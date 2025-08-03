"""
Audio-to-Text RAG for Podcast Search - Streamlit Application

A modern web interface for searching and analyzing podcast content using RAG.
"""

import streamlit as st
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from src.rag_engine import RAGEngine, create_rag_engine
from src.utils import (
    format_timestamp, format_duration, validate_audio_file,
    format_search_results, create_search_summary, load_config,
    create_directory_structure, get_audio_files_in_directory
)

# Page configuration
st.set_page_config(
    page_title="Podcast RAG Search",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .timestamp {
        color: #1f77b4;
        font-weight: bold;
    }
    .speaker {
        color: #ff7f0e;
        font-weight: bold;
    }
    .similarity {
        color: #2ca02c;
        font-weight: bold;
    }
    .voice-recorder {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    .recording-active {
        border-color: #e74c3c;
        background-color: #fdf2f2;
    }
    .microphone-icon {
        font-size: 2rem;
        color: #6c757d;
    }
    .microphone-icon.recording {
        color: #e74c3c;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}
if 'search_history' not in st.session_state:
    st.session_state.search_history = []


def initialize_rag_engine():
    """Initialize the RAG engine."""
    try:
        # Load configuration
        config = load_config()
        
        # Create RAG engine
        rag_engine = create_rag_engine(config)
        st.session_state.rag_engine = rag_engine
        
        return True
    except Exception as e:
        st.error(f"Error initializing RAG engine: {e}")
        return False


def process_audio_file(uploaded_file):
    """Process an uploaded audio file."""
    if st.session_state.rag_engine is None:
        st.error("RAG engine not initialized")
        return None
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path("data/audio")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate file
        is_valid, error_msg = validate_audio_file(str(temp_path))
        if not is_valid:
            st.error(f"Invalid audio file: {error_msg}")
            return None
        
        # Process the file
        with st.spinner("Processing audio file..."):
            result = st.session_state.rag_engine.process_audio_file(str(temp_path))
        
        # Save processing log
        st.session_state.processing_status[result["episode_id"]] = result
        
        return result
        
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None


def display_processing_results(result):
    """Display processing results."""
    if result is None:
        return
    
    st.success(f"✅ Successfully processed: {result['episode_id']}")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Duration",
            format_duration(result["transcription"]["duration"])
        )
    
    with col2:
        st.metric(
            "Segments",
            result["transcription"]["segments"]
        )
    
    with col3:
        st.metric(
            "Chunks Created",
            result["text_processing"]["chunks_created"]
        )
    
    with col4:
        st.metric(
            "Processing Time",
            f"{result['processing_time']:.1f}s"
        )
    
    # Show details in expander
    with st.expander("Processing Details"):
        st.json(result)


def search_podcasts(query, n_results=10, filter_episode=None, filter_speaker=None):
    """Search podcasts and return results."""
    if st.session_state.rag_engine is None:
        st.error("RAG engine not initialized")
        return None
    
    try:
        with st.spinner("Searching podcasts..."):
            results = st.session_state.rag_engine.search_podcasts(
                query, n_results, filter_episode, filter_speaker
            )
        
        # Add to search history
        st.session_state.search_history.append({
            "query": query,
            "timestamp": time.time(),
            "results_count": results["total_results"]
        })
        
        return results
        
    except Exception as e:
        st.error(f"Error searching podcasts: {e}")
        return None


def display_search_results(results):
    """Display search results in a beautiful format."""
    if results is None or not results["results"]:
        st.warning("No results found")
        return
    
    # Create summary
    summary = create_search_summary(results)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Results", summary["total_results"])
    
    with col2:
        st.metric("Episodes Found", summary["episodes_found"])
    
    with col3:
        st.metric("Speakers Found", summary["speakers_found"])
    
    with col4:
        st.metric("Avg Similarity", f"{summary['avg_similarity']:.3f}")
    
    # Display generated response if available
    if results.get("generated_response"):
        st.subheader("🤖 AI Generated Response")
        st.info(results["generated_response"])
    
    # Display individual results
    st.subheader("📋 Search Results")
    
    # Format results for display
    formatted_results = format_search_results(results["results"])
    
    for result in formatted_results:
        with st.container():
            st.markdown(f"""
            <div class="result-card">
                <h4>Rank #{result['rank']} - <span class="similarity">Similarity: {result['similarity_score']}</span></h4>
                <p><strong>Episode:</strong> {result['episode_id']}</p>
                <p><strong>Speaker:</strong> <span class="speaker">{result['speaker']}</span></p>
                <p><strong>Timestamp:</strong> <span class="timestamp">{result['timestamp']}</span> ({result['duration']})</p>
                <p><strong>Text:</strong> {result['text']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show full text in expander
            with st.expander("Show full text"):
                st.text(result["full_text"])


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🎙️ Podcast RAG Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Search and analyze podcast content using AI-powered RAG</p>', unsafe_allow_html=True)
    
    # Initialize RAG engine
    if st.session_state.rag_engine is None:
        if not initialize_rag_engine():
            st.error("Failed to initialize RAG engine. Please check your configuration.")
            return
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📁 Upload Audio", "🔍 Search", "📊 Analytics", "⚙️ Settings"]
    )
    
    # Home page
    if page == "🏠 Home":
        st.header("Welcome to Podcast RAG Search!")
        
        # Get database stats
        stats = st.session_state.rag_engine.get_database_stats()
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Episodes", stats.get("total_episodes", 0))
        
        with col2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        
        with col3:
            st.metric("Total Words", f"{stats.get('total_words', 0):,}")
        
        with col4:
            st.metric("Total Duration", format_duration(stats.get("total_duration", 0)))
        
        # Quick search
        st.subheader("🔍 Quick Search")
        
        # Quick search tabs
        quick_text_tab, quick_voice_tab = st.tabs(["⌨️ Text", "🎤 Voice"])
        
        with quick_text_tab:
            quick_query = st.text_input("Enter your search query:")
            if st.button("Search"):
                if quick_query:
                    results = search_podcasts(quick_query)
                    display_search_results(results)
        
        with quick_voice_tab:
            st.write("🎤 Voice recording is currently being updated. Please use the text search for now.")
            st.info("Voice recording feature will be available in the next update.")
            
            # Alternative: File upload for voice queries
            voice_file = st.file_uploader(
                "Or upload an audio file with your search query:",
                type=['wav', 'mp3', 'm4a'],
                key="voice_query_upload"
            )
            
            if voice_file:
                st.audio(voice_file, format="audio/wav")
                
                if st.button("Process Voice Query"):
                    # Create temporary file for voice query
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(voice_file.getbuffer())
                        tmp_voice_path = tmp_file.name
                    
                    try:
                        with st.spinner("Converting voice to text..."):
                            # Use the audio processor to transcribe the voice query
                            transcription = st.session_state.rag_engine.audio_processor.transcribe_audio(tmp_voice_path)
                            
                            if transcription and transcription.get("text"):
                                quick_query = transcription["text"]
                                st.success(f"Voice query: '{quick_query}'")
                                results = search_podcasts(quick_query)
                                display_search_results(results)
                            else:
                                st.error("Could not understand voice query. Please try again.")
                                
                    except Exception as e:
                        st.error(f"Error processing voice query: {e}")
                    finally:
                        # Clean up temporary file
                        import os
                        if os.path.exists(tmp_voice_path):
                            os.unlink(tmp_voice_path)
        
        # Recent activity
        if st.session_state.search_history:
            st.subheader("📈 Recent Searches")
            for search in st.session_state.search_history[-5:]:
                st.text(f"'{search['query']}' - {search['results_count']} results")
    
    # Upload Audio page
    elif page == "📁 Upload Audio":
        st.header("Upload Audio Files")
        
        # Create tabs for different upload methods
        upload_tab, record_tab = st.tabs(["📁 File Upload", "🎤 Record Audio"])
        
        with upload_tab:
            # File upload
            uploaded_files = st.file_uploader(
                "Choose audio files",
                type=['mp3', 'wav', 'm4a', 'flac'],
                accept_multiple_files=True
            )
        
        with record_tab:
            st.subheader("🎤 Record Audio")
            st.write("Record audio directly from your microphone")
            
            # Note about voice recording
            st.info("🎤 Voice recording feature is being updated. For now, you can:")
            
            # Option 1: Upload recorded audio
            st.write("**Option 1: Upload recorded audio file**")
            recorded_audio = st.file_uploader(
                "Upload your recorded audio:",
                type=['wav', 'mp3', 'm4a', 'flac'],
                key="recorded_audio_upload"
            )
            
            if recorded_audio:
                st.audio(recorded_audio, format="audio/wav")
                
                # Process the uploaded recorded audio
                if st.button("Save and Process Recording"):
                    # Create a temporary file for the recorded audio
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(recorded_audio.getbuffer())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Process the recorded audio
                        with st.spinner("Processing recorded audio..."):
                            result = st.session_state.rag_engine.process_audio_file(tmp_file_path)
                            
                        if result:
                            st.success("Recording processed successfully!")
                            display_processing_results(result)
                        else:
                            st.error("Failed to process recording")
                            
                    except Exception as e:
                        st.error(f"Error processing recording: {e}")
                    finally:
                        # Clean up temporary file
                        import os
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
            
            # Option 2: Instructions for manual recording
            st.write("**Option 2: Manual recording instructions**")
            st.write("1. Use your device's voice recorder app")
            st.write("2. Record your audio content")
            st.write("3. Save as WAV, MP3, or M4A format")
            st.write("4. Upload the file above")
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} file(s)")
            
            # Process files
            if st.button("Process Audio Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    result = process_audio_file(uploaded_file)
                    if result:
                        display_processing_results(result)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                st.success("All files processed successfully!")
        
        # Batch processing from directory
        st.subheader("Batch Processing")
        directory_path = st.text_input("Enter directory path for batch processing:")
        if st.button("Process Directory"):
            if directory_path and os.path.exists(directory_path):
                audio_files = get_audio_files_in_directory(directory_path)
                if audio_files:
                    st.write(f"Found {len(audio_files)} audio files")
                    
                    if st.button("Start Batch Processing"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, audio_file in enumerate(audio_files):
                            status_text.text(f"Processing {Path(audio_file).name}...")
                            
                            try:
                                result = st.session_state.rag_engine.process_audio_file(audio_file)
                                if result:
                                    st.session_state.processing_status[result["episode_id"]] = result
                            except Exception as e:
                                st.error(f"Error processing {audio_file}: {e}")
                            
                            progress_bar.progress((i + 1) / len(audio_files))
                        
                        status_text.text("Batch processing complete!")
                        st.success(f"Processed {len(audio_files)} files!")
                else:
                    st.warning("No audio files found in directory")
            else:
                st.error("Directory not found")
    
    # Search page
    elif page == "🔍 Search":
        st.header("Search Podcasts")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create tabs for text and voice search
            search_tab, voice_tab = st.tabs(["⌨️ Text Search", "🎤 Voice Search"])
            
            with search_tab:
                query = st.text_input("Enter your search query:", placeholder="e.g., machine learning, climate change, startup funding")
            
            with voice_tab:
                st.write("Record your search query")
                st.info("🎤 Voice recording is being updated. Please upload an audio file with your search query:")
                
                voice_audio = st.file_uploader(
                    "Upload audio file with your search query:",
                    type=['wav', 'mp3', 'm4a'],
                    key="voice_search_upload"
                )
                
                if voice_audio:
                    st.audio(voice_audio, format="audio/wav")
                    
                    if st.button("Process Voice Query"):
                        # Create temporary file for voice query
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(voice_audio.getbuffer())
                            tmp_voice_path = tmp_file.name
                        
                        try:
                            with st.spinner("Converting voice to text..."):
                                # Use the audio processor to transcribe the voice query
                                transcription = st.session_state.rag_engine.audio_processor.transcribe_audio(tmp_voice_path)
                                
                                if transcription and transcription.get("text"):
                                    query = transcription["text"]
                                    st.success(f"Voice query: '{query}'")
                                    st.session_state.voice_query = query
                                else:
                                    st.error("Could not understand voice query. Please try again.")
                                    
                        except Exception as e:
                            st.error(f"Error processing voice query: {e}")
                        finally:
                            # Clean up temporary file
                            import os
                            if os.path.exists(tmp_voice_path):
                                os.unlink(tmp_voice_path)
                
                # Display processed voice query
                if hasattr(st.session_state, 'voice_query'):
                    st.text_input("Voice Query:", value=st.session_state.voice_query, key="voice_input")
                    query = st.session_state.voice_query
        
        with col2:
            n_results = st.number_input("Number of results:", min_value=1, max_value=50, value=10)
        
        # Filters
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            episodes = st.session_state.rag_engine.list_episodes()
            filter_episode = st.selectbox("Filter by episode:", ["All episodes"] + episodes)
        
        with col2:
            # Get unique speakers from database
            stats = st.session_state.rag_engine.get_database_stats()
            speakers = ["All speakers"]  # Placeholder - would need to extract from database
            filter_speaker = st.selectbox("Filter by speaker:", speakers)
        
        # Search button
        if st.button("🔍 Search", type="primary"):
            if query:
                # Apply filters
                episode_filter = None if filter_episode == "All episodes" else filter_episode
                speaker_filter = None if filter_speaker == "All speakers" else filter_speaker
                
                results = search_podcasts(query, n_results, episode_filter, speaker_filter)
                display_search_results(results)
            else:
                st.warning("Please enter a search query")
        
        # Search examples
        st.subheader("💡 Search Examples")
        examples = [
            "What did they say about artificial intelligence?",
            "Find discussions about climate change",
            "Show me segments about startup funding",
            "What are the main points about AI ethics?",
            "Find conversations about renewable energy"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.example_query = example
                st.rerun()
        
        # Check if example was selected
        if hasattr(st.session_state, 'example_query'):
            query = st.session_state.example_query
            st.text_input("Query:", value=query, key="example_input")
            if st.button("Search Example"):
                results = search_podcasts(query, n_results)
                display_search_results(results)
            del st.session_state.example_query
    
    # Analytics page
    elif page == "📊 Analytics":
        st.header("Analytics & Insights")
        
        # Database statistics
        stats = st.session_state.rag_engine.get_database_stats()
        
        st.subheader("📈 Database Statistics")
        
        # Create metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Episodes", stats.get("total_episodes", 0))
        
        with col2:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
        
        with col3:
            st.metric("Total Words", f"{stats.get('total_words', 0):,}")
        
        with col4:
            st.metric("Total Duration", format_duration(stats.get("total_duration", 0)))
        
        # Episode details
        st.subheader("📋 Episode Details")
        episodes = st.session_state.rag_engine.list_episodes()
        
        if episodes:
            selected_episode = st.selectbox("Select episode:", episodes)
            if selected_episode:
                episode_stats = st.session_state.rag_engine.get_episode_summary(selected_episode)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Chunks", episode_stats.get("total_chunks", 0))
                
                with col2:
                    st.metric("Words", episode_stats.get("total_words", 0))
                
                with col3:
                    st.metric("Duration", format_duration(episode_stats.get("total_duration", 0)))
                
                # Speakers
                if episode_stats.get("speakers"):
                    st.write("**Speakers:**", ", ".join(episode_stats["speakers"]))
        
        # Search history
        if st.session_state.search_history:
            st.subheader("🔍 Search History")
            
            # Convert to DataFrame for better display
            history_df = pd.DataFrame(st.session_state.search_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')
            history_df = history_df.sort_values('timestamp', ascending=False)
            
            st.dataframe(history_df, use_container_width=True)
    
    # Settings page
    elif page == "⚙️ Settings":
        st.header("Settings")
        
        # Configuration
        st.subheader("🔧 Configuration")
        
        # Load current config
        config = load_config()
        
        # Model settings
        col1, col2 = st.columns(2)
        
        with col1:
            whisper_model = st.selectbox(
                "Whisper Model:",
                ["tiny", "base", "small", "medium", "large"],
                index=["tiny", "base", "small", "medium", "large"].index(config.get("whisper_model", "base"))
            )
        
        with col2:
            embedding_model = st.selectbox(
                "Embedding Model:",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-v2"],
                index=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-v2"].index(config.get("embedding_model", "all-MiniLM-L6-v2"))
            )
        
        # Processing settings
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input(
                "Chunk Size:",
                min_value=100,
                max_value=2000,
                value=config.get("chunk_size", 1000),
                step=100
            )
        
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap:",
                min_value=0,
                max_value=500,
                value=config.get("chunk_overlap", 200),
                step=50
            )
        
        # Save configuration
        if st.button("Save Configuration"):
            new_config = {
                "whisper_model": whisper_model,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "collection_name": config.get("collection_name", "podcast_episodes"),
                "persist_directory": config.get("persist_directory", "./data/database")
            }
            
            # Save config
            import json
            with open("config.json", "w") as f:
                json.dump(new_config, f, indent=2)
            
            st.success("Configuration saved!")
            st.info("Please restart the application for changes to take effect.")
        
        # Database management
        st.subheader("🗄️ Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Data"):
                export_path = "exports/podcast_data_export.json"
                os.makedirs("exports", exist_ok=True)
                
                if st.session_state.rag_engine.export_data(export_path):
                    st.success(f"Data exported to {export_path}")
                else:
                    st.error("Failed to export data")
        
        with col2:
            if st.button("Clear Search History"):
                st.session_state.search_history = []
                st.success("Search history cleared!")
        
        # System information
        st.subheader("ℹ️ System Information")
        
        import platform
        import torch
        
        system_info = {
            "Python Version": platform.python_version(),
            "Platform": platform.platform(),
            "PyTorch Version": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "Database Path": config.get("persist_directory", "./data/database")
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")


if __name__ == "__main__":
    # Create directory structure
    create_directory_structure()
    
    # Run the main application
    main() 