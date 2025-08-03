# Audio-to-Text RAG for Podcast Search

A multimodal RAG (Retrieval-Augmented Generation) system that processes audio podcasts, converts them to searchable text, and allows users to query specific topics mentioned across multiple episodes with timestamp references.

## 🎯 Features

- **High-accuracy audio-to-text conversion** using Whisper
- **Multi-episode search** across podcast libraries
- **Topic-based querying** with contextual understanding
- **Timestamp referencing** for precise audio segment location
- **Speaker identification and diarization** for multi-speaker podcasts
- **Vector-based semantic search** using sentence transformers
- **Clean web interface** with Streamlit

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Whisper STT    │───▶│  Text Chunks    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │◀───│  Vector Search  │◀───│  Embeddings     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LLM Response   │    │  Timestamp Ref  │    │  Chroma DB      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- OpenAI API key (optional, for enhanced responses)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd audio-to-text-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
audio-to-text-rag/
├── app.py                 # Main Streamlit application
├── src/
│   ├── audio_processor.py # Audio preprocessing and STT
│   ├── text_processor.py  # Text chunking and processing
│   ├── embeddings.py      # Vector embeddings and search
│   ├── rag_engine.py      # RAG pipeline orchestration
│   └── utils.py           # Utility functions
├── data/
│   ├── audio/             # Audio files storage
│   ├── transcripts/       # Generated transcripts
│   └── database/          # Vector database
├── notebooks/
│   └── evaluation.ipynb   # System evaluation
├── tests/
│   └── test_*.py          # Unit tests
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI API (optional, for enhanced responses)
OPENAI_API_KEY=your_openai_api_key

# HuggingFace (for embeddings)
HUGGINGFACE_API_KEY=your_huggingface_token

# Model configurations
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Model Options

- **Whisper Models**: `tiny`, `base`, `small`, `medium`, `large`
- **Embedding Models**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `multi-qa-MiniLM-L6-v2`

## 📊 Usage

### 1. Upload Audio Files

- Supported formats: MP3, WAV, M4A, FLAC
- Maximum file size: 100MB per file
- Batch upload supported

### 2. Process Episodes

- Automatic audio-to-text conversion
- Speaker diarization (if multiple speakers detected)
- Timestamp generation for each segment
- Vector embedding creation

### 3. Search and Query

- Natural language queries
- Topic-based search across all episodes
- Timestamp references for audio segments
- Relevance scoring

### 4. Example Queries

- "What did they say about machine learning?"
- "Find discussions about climate change"
- "Show me segments about startup funding"
- "What are the main points about AI ethics?"

## 🧪 Evaluation

The system includes evaluation metrics:

- **Retrieval Accuracy**: Precision@K, Recall@K
- **Response Relevance**: RAGAS metrics
- **Latency**: Query response time
- **Audio Quality**: Transcription accuracy

Run evaluation:
```bash
python -m pytest tests/
jupyter notebook notebooks/evaluation.ipynb
```

## 🔍 Technical Details

### Audio Processing Pipeline

1. **Preprocessing**: Noise reduction, normalization
2. **STT Conversion**: OpenAI Whisper for transcription
3. **Speaker Diarization**: Pyannote.audio for speaker identification
4. **Timestamp Mapping**: Precise time alignment

### Text Processing

1. **Chunking**: Semantic-aware text segmentation
2. **Embedding**: Sentence transformers for vector representation
3. **Indexing**: Chroma vector database for fast retrieval

### RAG Pipeline

1. **Query Processing**: Embedding generation
2. **Retrieval**: Vector similarity search
3. **Reranking**: Relevance scoring
4. **Generation**: Context-aware response generation

## 🛠️ Development

### Adding New Features

1. **Custom Embeddings**: Modify `src/embeddings.py`
2. **Audio Processing**: Extend `src/audio_processor.py`
3. **UI Components**: Update `app.py`

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_audio_processor.py
```

## 📈 Performance Optimization

- **Batch Processing**: Parallel audio processing
- **Caching**: Embedding and transcription caching
- **Indexing**: Efficient vector database queries
- **Streaming**: Real-time audio processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenAI for Whisper
- HuggingFace for sentence transformers
- Chroma for vector database
- Streamlit for the web interface

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the evaluation notebook

---

**Demo Links:**
- [Live Application](https://your-app-url.streamlit.app)
- [GitHub Repository](https://github.com/your-username/audio-to-text-rag) 