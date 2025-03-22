# Auto detect text files and perform LF normalization
* text=auto

# Python files
*.py text diff=python

# JavaScript files
*.js text

# HTML files
*.html text diff=html

# CSS files
*.css text diff=css

# Markdown files
*.md text diff=markdown

# Data files
*.json text
*.csv text

# Audio files
*.wav binary
*.mp3 binary

# Image files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary

# Ignore all log files
*.log -text

# Streamlit cache
.streamlit/ export-ignore

# Python cache
__pycache__/ export-ignore
*.py[cod] export-ignore
*$py.class export-ignore

# Distribution / packaging
dist/ export-ignore
build/ export-ignore
*.egg-info/ export-ignore
