import streamlit as st
import requests
import json
import base64
from io import BytesIO
import threading
import time
import os
import sys
import subprocess
import multiprocessing

# Import API server starter function
try:
    from api import start_api_server
except ImportError:
    # If import fails, try to find api.py in the same directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    try:
        from api import start_api_server
    except ImportError:
        st.error("Cannot import api.py. Make sure it's in the same directory.")
        start_api_server = None

# Set page configuration
st.set_page_config(
    page_title="News Sentiment Analysis & TTS",
    page_icon="📰",
    layout="wide"
)

# App title and description
st.title("📰 News Sentiment Analysis & Hindi TTS")
st.markdown("""
This application extracts key details from news articles related to a given company, 
performs sentiment analysis, conducts comparative analysis, and generates a text-to-speech output in Hindi.
""")

# Base API URL (adjust for deployment)
API_URL = "http://localhost:8000"

# Add a status indicator
api_status = st.empty()

# Start API server if it's not already running
def ensure_api_running():
    # First try to check if API is already running
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            api_status.success("API is running and ready.")
            return True
    except:
        api_status.warning("API not detected. Starting API server...")
    
    # If API is not running, start it
    if start_api_server:
        try:
            api_thread = start_api_server()
            # Check if API is now running
            for _ in range(5):  # Try 5 times with small delays
                try:
                    response = requests.get(f"{API_URL}/", timeout=1)
                    if response.status_code == 200:
                        api_status.success("API is now running and ready.")
                        return True
                except:
                    time.sleep(1)
            
            # If we reach here, API didn't start properly
            api_status.error("Failed to start API. Check the console for errors.")
        except Exception as e:
            # Try starting API as a subprocess instead
            api_status.warning(f"Error starting API in thread: {str(e)}. Trying subprocess...")
            try:
                # Get the directory of the current script
                current_dir = os.path.dirname(os.path.abspath(__file__))
                api_path = os.path.join(current_dir, "api.py")
                
                # Start the API in a subprocess
                if sys.platform.startswith('win'):
                    # Windows
                    subprocess.Popen(
                        [sys.executable, api_path],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    # Linux/Mac
                    subprocess.Popen(
                        [sys.executable, api_path],
                        start_new_session=True
                    )
                
                # Wait for API to start
                time.sleep(5)
                
                # Check if API is now running
                try:
                    response = requests.get(f"{API_URL}/", timeout=2)
                    if response.status_code == 200:
                        api_status.success("API is now running and ready (started via subprocess).")
                        return True
                    else:
                        api_status.error("API started but is not responding correctly.")
                except:
                    api_status.error("Failed to start API via subprocess.")
            except Exception as sub_e:
                api_status.error(f"Failed to start API: {str(sub_e)}")
    else:
        api_status.error("Cannot start API. Make sure api.py is in the same directory.")
    
    return False

# Try to ensure API is running
api_running = ensure_api_running()

# Company input
st.subheader("Enter Company Name")
company_name = st.text_input("Company Name", value="Tesla")

# Add example companies for quick selection
examples = st.expander("Example Companies")
example_companies = ["Tesla", "Apple", "Google", "Microsoft", "Amazon", "Meta", "Netflix"]
example_cols = examples.columns(len(example_companies))
for i, example in enumerate(example_companies):
    if example_cols[i].button(example):
        company_name = example

# Form submission
if st.button("Analyze News"):
    if company_name:
        # Check API again if it wasn't running before
        if not api_running:
            api_running = ensure_api_running()
            
        if not api_running:
            st.error("Cannot analyze news because the API is not running.")
        else:
            with st.spinner(f"Fetching and analyzing news for {company_name}... This may take up to 3 minutes for comprehensive analysis."):
                try:
                    # Call API for news analysis with increased timeout (3 minutes)
                    response = requests.post(
                        f"{API_URL}/analyze",
                        json={"company_name": company_name},
                        timeout=180  # Increased to 3 minutes (180 seconds)
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display company info
                        st.subheader(f"Analysis Results for {data['Company']}")
                        
                        # Display articles in expandable sections
                        st.write("### News Articles")
                        if not data["Articles"]:
                            st.warning("No articles found. Displaying sample data for demonstration.")
                        
                        # Count sentiments for verification
                        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
                        
                        for i, article in enumerate(data['Articles'], 1):
                            # Count sentiments
                            sentiment_counts[article['Sentiment']] += 1
                            
                            with st.expander(f"Article {i}: {article['Title']}"):
                                st.write(f"**Summary:** {article['Summary']}")
                                
                                # Color-code sentiment
                                sentiment = article['Sentiment']
                                if sentiment == "Positive":
                                    st.markdown(f"**Sentiment:** <span style='color:green'>{sentiment}</span>", unsafe_allow_html=True)
                                elif sentiment == "Negative":
                                    st.markdown(f"**Sentiment:** <span style='color:red'>{sentiment}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"**Sentiment:** <span style='color:gray'>{sentiment}</span>", unsafe_allow_html=True)
                                
                                st.write("**Topics:**", ", ".join(article['Topics']))
                        
                        # Display sentiment count verification
                        st.write("### Sentiment Distribution")
                        st.write(f"📊 Actual sentiment counts in articles: Positive: {sentiment_counts['Positive']}, Negative: {sentiment_counts['Negative']}, Neutral: {sentiment_counts['Neutral']}")
                        
                        # Display comparative analysis
                        st.write("### Comparative Sentiment Analysis")
                        
                        # Sentiment distribution
                        sentiment_dist = data['Comparative Sentiment Score']['Sentiment Distribution']
                        
                        # Create a bar chart for sentiment distribution
                        st.bar_chart({
                            'Positive': [sentiment_dist['Positive']],
                            'Neutral': [sentiment_dist['Neutral']],
                            'Negative': [sentiment_dist['Negative']]
                        })
                        
                        # Display coverage differences
                        st.write("#### Coverage Differences")
                        if data['Comparative Sentiment Score']['Coverage Differences']:
                            for comparison in data['Comparative Sentiment Score']['Coverage Differences']:
                                st.write(f"- **Comparison:** {comparison['Comparison']}")
                                st.write(f"  **Impact:** {comparison['Impact']}")
                        else:
                            st.info("Not enough articles for comparative analysis.")
                        
                        # Display topic overlap
                        st.write("#### Topic Analysis")
                        topic_overlap = data['Comparative Sentiment Score']['Topic Overlap']
                        
                        if topic_overlap['Common Topics']:
                            st.write(f"**Common Topics:** {', '.join(topic_overlap['Common Topics'])}")
                        else:
                            st.write("**Common Topics:** None found")
                        
                        # Get unique topics from all articles
                        all_unique_topics = []
                        for key, value in topic_overlap.items():
                            if key.startswith("Unique Topics in Article"):
                                all_unique_topics.extend(value)
                        
                        if all_unique_topics:
                            st.write(f"**Unique Topics:** {', '.join(set(all_unique_topics))}")
                        else:
                            st.write("**Unique Topics:** None found")
                        
                        # Display final sentiment
                        st.write("### Final Sentiment Analysis")
                        st.write(data['Final Sentiment Analysis'])
                        
                        # Audio playback
                        st.write("### Hindi TTS Summary")
                        if 'audio_base64' in data and data['audio_base64']:
                            audio_bytes = base64.b64decode(data['audio_base64'])
                            st.audio(audio_bytes, format="audio/wav")
                        else:
                            st.error("Audio generation failed or not available")
                    
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The analysis may be taking longer than expected. Try a different company or try again later.")
                
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the API server. Please ensure the API server is running at localhost:8000.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a company name")

# Run this when the script is executed directly
if __name__ == "__main__":
    # The API starting logic is at the beginning of the script
    pass