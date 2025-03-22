# News Sentiment Analysis & Hindi TTS

This application extracts news articles related to a company, performs sentiment analysis, conducts comparative analysis, and generates text-to-speech output in Hindi.

## Features

- **News Article Extraction**: Gathers news articles from various sources including financial news sites, social media, and search engines
- **Sentiment Analysis**: Analyzes the sentiment of each article (Positive, Negative, or Neutral)
- **Topic Extraction**: Identifies key topics from article content
- **Comparative Analysis**: Compares sentiment across articles and identifies common/unique topics
- **Hindi Text-to-Speech**: Generates audio summaries in Hindi
- **Interactive UI**: User-friendly Streamlit interface for easy interaction

## Architecture

The application consists of three main components:

1. **FastAPI Backend** (`api.py`): Handles requests, processes data, and returns analysis results
2. **Streamlit Frontend** (`app.py`): Provides the user interface for interacting with the application
3. **Utility Functions** (`utils.py`): Contains the core functionality for news extraction, sentiment analysis, and other processing tasks

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/InfinityM10/News-Sentiment-Analysis-and--Translation.git
   cd news-sentiment-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install Ollama for improved sentiment analysis and summarization:
   - Follow installation instructions at [Ollama](https://ollama.ai/)
   - Pull the Llama3 model:
     ```
     ollama pull llama3
     ```

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

3. Enter a company name in the input field and click "Analyze News"

4. View the analysis results including:
   - News article summaries
   - Sentiment distribution
   - Comparative analysis
   - Hindi TTS audio summary

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /`: Check if the API is running
- `POST /analyze`: Analyze news for a given company
  - Request Body: `{"company_name": "Company Name"}`
  - Returns: Analysis results including articles, sentiment scores, and TTS audio

## Technical Details

- **News Extraction**: Uses multiple sources and rotating user agents to avoid rate limiting
- **Sentiment Analysis**: Employs a hybrid approach using keyword analysis, transformer models, and Llama3 when available
- **Topic Extraction**: Utilizes Llama3 for intelligent topic extraction with fallback to frequency-based extraction
- **Hindi TTS**: Uses Google's Text-to-Speech (gTTS) API for generating Hindi audio

## Requirements

- Python 3.8+
- FastAPI
- Streamlit
- BeautifulSoup4
- Transformers
- gTTS
- Other dependencies listed in `requirements.txt`

## Optional Components

- **Ollama**: For improved sentiment analysis and summarization using Llama3
