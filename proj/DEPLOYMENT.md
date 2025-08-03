# Deployment Guide - Audio-to-Text RAG for Podcast Search

This guide provides step-by-step instructions for deploying the podcast RAG system on various platforms.

## 🚀 Quick Deployment Options

### 1. Local Development

#### Prerequisites
- Python 3.8+
- FFmpeg installed
- Git

#### Steps
```bash
# Clone the repository
git clone <your-repo-url>
cd audio-to-text-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys

# Run the application
streamlit run app.py
```

### 2. Streamlit Cloud Deployment

#### Steps
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set deployment settings:
     - Python version: 3.8
     - Main file path: `app.py`
   - Add secrets in the dashboard:
     ```
     OPENAI_API_KEY = your_openai_api_key
     HUGGINGFACE_API_KEY = your_huggingface_token
     ```

3. **Configure Streamlit**
   Create `streamlit_config.toml`:
   ```toml
   [server]
   maxUploadSize = 100
   enableXsrfProtection = false
   
   [browser]
   gatherUsageStats = false
   ```

### 3. Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/audio data/transcripts data/database data/processed

# Expose port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "app.py"]
```

#### Build and Run
```bash
# Build image
docker build -t podcast-rag .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e HUGGINGFACE_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  podcast-rag
```

### 4. Heroku Deployment

#### Create Procfile
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

#### Create runtime.txt
```
python-3.9.18
```

#### Deploy
```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key
heroku config:set HUGGINGFACE_API_KEY=your_key

# Deploy
git push heroku main
```

### 5. Google Cloud Run

#### Create Dockerfile (same as above)

#### Deploy
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/podcast-rag

# Deploy to Cloud Run
gcloud run deploy podcast-rag \
  --image gcr.io/PROJECT_ID/podcast-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key,HUGGINGFACE_API_KEY=your_key
```

## 🔧 Configuration

### Environment Variables
```bash
# Required for enhanced responses
OPENAI_API_KEY=your_openai_api_key

# Required for HuggingFace models
HUGGINGFACE_API_KEY=your_huggingface_token

# Model configurations
WHISPER_MODEL=base
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Database settings
CHROMA_PERSIST_DIRECTORY=./data/database
COLLECTION_NAME=podcast_episodes

# Application settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Model Selection Guide

#### Whisper Models
- **tiny**: Fastest, lowest accuracy (39M parameters)
- **base**: Good balance (74M parameters) - **Recommended**
- **small**: Better accuracy (244M parameters)
- **medium**: High accuracy (769M parameters)
- **large**: Best accuracy (1550M parameters)

#### Embedding Models
- **all-MiniLM-L6-v2**: Fast, good quality (384 dimensions)
- **all-mpnet-base-v2**: Better quality, slower (768 dimensions)
- **multi-qa-MiniLM-L6-v2**: Optimized for Q&A (384 dimensions)

## 📊 Performance Optimization

### For Production Deployment

1. **Use Larger Models**
   ```bash
   WHISPER_MODEL=medium
   EMBEDDING_MODEL=all-mpnet-base-v2
   ```

2. **Optimize Chunking**
   ```bash
   CHUNK_SIZE=800
   CHUNK_OVERLAP=150
   ```

3. **Enable Caching**
   ```bash
   ENABLE_CACHING=True
   CACHE_TTL=3600
   ```

4. **Database Optimization**
   - Use persistent storage for ChromaDB
   - Consider using external vector database (Pinecone, Weaviate)

### Resource Requirements

#### Minimum (Development)
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB

#### Recommended (Production)
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+
- GPU: Optional (for faster processing)

## 🔒 Security Considerations

### API Key Management
- Never commit API keys to version control
- Use environment variables or secret management
- Rotate keys regularly

### File Upload Security
- Validate file types and sizes
- Scan uploaded files for malware
- Implement rate limiting

### Data Privacy
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement user authentication if needed

## 📈 Monitoring and Logging

### Enable Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks
Create `health_check.py`:
```python
import requests
import sys

def health_check():
    try:
        response = requests.get('http://localhost:8501/_stcore/health')
        if response.status_code == 200:
            print("✅ Application is healthy")
            return True
        else:
            print("❌ Application health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    success = health_check()
    sys.exit(0 if success else 1)
```

## 🚨 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **CUDA/GPU issues**
   ```bash
   # Install CPU-only PyTorch
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory issues**
   - Reduce chunk size
   - Use smaller models
   - Process files in batches

4. **Slow performance**
   - Use GPU if available
   - Optimize chunk size
   - Use faster models for development

### Debug Mode
```bash
# Enable debug logging
export DEBUG_MODE=True
streamlit run app.py --logger.level=debug
```

## 📞 Support

For deployment issues:
1. Check the logs: `docker logs <container_id>`
2. Verify environment variables
3. Test locally first
4. Check system requirements

## 🔄 Updates and Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the application
# For Docker: docker-compose restart
# For systemd: sudo systemctl restart podcast-rag
```

### Backup Strategy
```bash
# Backup database
cp -r data/database backup/database_$(date +%Y%m%d)

# Backup transcripts
cp -r data/transcripts backup/transcripts_$(date +%Y%m%d)
```

### Monitoring Script
```bash
#!/bin/bash
# monitor.sh
while true; do
    if ! curl -f http://localhost:8501/_stcore/health; then
        echo "Application down, restarting..."
        docker-compose restart
    fi
    sleep 60
done
```

---

**Next Steps:**
1. Choose your deployment method
2. Set up environment variables
3. Test with sample audio files
4. Monitor performance and adjust configuration
5. Set up backup and monitoring 