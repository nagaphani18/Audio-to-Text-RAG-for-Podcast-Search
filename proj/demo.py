"""
Demo script for Audio-to-Text RAG for Podcast Search

This script demonstrates the core functionality of the system.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_engine import RAGEngine, create_rag_engine
from src.utils import create_directory_structure, load_config


def create_sample_transcription():
    """Create a sample transcription for demonstration."""
    sample_data = {
        "episode_id": "demo_episode_001",
        "audio_path": "sample_audio.mp3",
        "transcription": "This is a sample podcast episode about artificial intelligence and machine learning. We discuss the latest developments in AI technology and their impact on society. The conversation covers topics like deep learning, neural networks, and the future of automation.",
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a sample podcast episode about artificial intelligence and machine learning.",
                "speaker": "Speaker_1",
                "confidence": 0.95
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": "We discuss the latest developments in AI technology and their impact on society.",
                "speaker": "Speaker_2",
                "confidence": 0.92
            },
            {
                "start": 10.0,
                "end": 15.0,
                "text": "The conversation covers topics like deep learning, neural networks, and the future of automation.",
                "speaker": "Speaker_1",
                "confidence": 0.88
            }
        ],
        "metadata": {
            "language": "en",
            "duration": 15.0,
            "transcription_time": 2.5,
            "total_segments": 3,
            "speakers": ["Speaker_1", "Speaker_2"]
        }
    }
    
    # Save sample transcription
    os.makedirs("data/transcripts", exist_ok=True)
    with open("data/transcripts/demo_episode_001.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    return sample_data


def create_sample_chunks():
    """Create sample chunks for demonstration."""
    sample_chunks = [
        {
            "id": "chunk_001",
            "text": "This is a sample podcast episode about artificial intelligence and machine learning.",
            "metadata": {
                "keywords": ["podcast", "artificial", "intelligence", "machine", "learning"],
                "word_count": 12,
                "sentence_count": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "speaker": "Speaker_1",
                "episode_id": "demo_episode_001",
                "confidence": 0.95
            }
        },
        {
            "id": "chunk_002",
            "text": "We discuss the latest developments in AI technology and their impact on society.",
            "metadata": {
                "keywords": ["developments", "technology", "impact", "society"],
                "word_count": 11,
                "sentence_count": 1,
                "start_time": 5.0,
                "end_time": 10.0,
                "speaker": "Speaker_2",
                "episode_id": "demo_episode_001",
                "confidence": 0.92
            }
        },
        {
            "id": "chunk_003",
            "text": "The conversation covers topics like deep learning, neural networks, and the future of automation.",
            "metadata": {
                "keywords": ["conversation", "deep", "learning", "neural", "networks", "automation"],
                "word_count": 13,
                "sentence_count": 1,
                "start_time": 10.0,
                "end_time": 15.0,
                "speaker": "Speaker_1",
                "episode_id": "demo_episode_001",
                "confidence": 0.88
            }
        }
    ]
    
    # Save sample chunks
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/demo_episode_001_chunks.json", "w") as f:
        json.dump(sample_chunks, f, indent=2)
    
    return sample_chunks


def run_demo():
    """Run the complete demo."""
    print("🎙️ Audio-to-Text RAG for Podcast Search - Demo")
    print("=" * 60)
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    print("✅ Directories created successfully")
    
    # Initialize RAG engine
    print("\n🔧 Initializing RAG engine...")
    try:
        config = load_config()
        rag_engine = create_rag_engine(config)
        print("✅ RAG engine initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG engine: {e}")
        return
    
    # Create sample data
    print("\n📝 Creating sample data...")
    sample_transcription = create_sample_transcription()
    sample_chunks = create_sample_chunks()
    print("✅ Sample data created successfully")
    
    # Add chunks to database
    print("\n🗄️ Adding chunks to database...")
    try:
        result = rag_engine.embedding_manager.add_chunks_to_database(sample_chunks)
        print(f"✅ Added {result['added']} chunks to database")
    except Exception as e:
        print(f"❌ Error adding chunks to database: {e}")
        return
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural networks",
        "automation"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        try:
            results = rag_engine.search_podcasts(query, n_results=5)
            print(f"Found {results['total_results']} results")
            
            if results['results']:
                print("Top result:")
                top_result = results['results'][0]
                print(f"  Text: {top_result['text'][:100]}...")
                print(f"  Similarity: {top_result['similarity_score']:.3f}")
                print(f"  Speaker: {top_result['metadata']['speaker']}")
                print(f"  Timestamp: {top_result['metadata']['start_time']:.1f}s")
            
        except Exception as e:
            print(f"❌ Error searching for '{query}': {e}")
    
    # Display database statistics
    print("\n📊 Database Statistics:")
    stats = rag_engine.get_database_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test episode summary
    print("\n📋 Episode Summary:")
    episode_summary = rag_engine.get_episode_summary("demo_episode_001")
    for key, value in episode_summary.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'streamlit run app.py' to start the web interface")
    print("2. Upload your own audio files")
    print("3. Search through your podcast content")
    print("4. Explore the analytics and insights")


def run_quick_test():
    """Run a quick functionality test."""
    print("🧪 Quick Functionality Test")
    print("=" * 40)
    
    # Test basic imports
    try:
        from src.utils import format_timestamp, format_duration
        print("✅ Basic imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test utility functions
    try:
        assert format_timestamp(61.5) == "01:01"
        assert format_duration(90) == "1.5m"
        print("✅ Utility functions working")
    except Exception as e:
        print(f"❌ Utility function error: {e}")
        return
    
    # Test configuration loading
    try:
        config = load_config()
        print("✅ Configuration loading successful")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Podcast RAG System Demo")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        run_demo() 