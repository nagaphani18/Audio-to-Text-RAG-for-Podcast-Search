"""
Demo script for testing voice recording functionality.
This script can be used to test the microphone features independently.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from src.audio_processor import AudioProcessor

def main():
    st.title("🎤 Voice Recording Demo")
    st.write("Test the microphone recording functionality")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(model_name="base")
    
    # Voice recording
    st.subheader("Upload Audio for Transcription")
    st.info("🎤 Upload an audio file to test transcription")
    
    audio_file = st.file_uploader(
        "Choose an audio file:",
        type=['wav', 'mp3', 'm4a', 'flac'],
        key="demo_audio_upload"
    )
    
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        
        if st.button("Transcribe Audio"):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            try:
                with st.spinner("Transcribing audio..."):
                    transcription = audio_processor.transcribe_audio(tmp_file_path)
                
                if transcription and transcription.get("text"):
                    st.success("Transcription successful!")
                    st.write("**Transcribed text:**")
                    st.write(transcription["text"])
                    
                    # Show additional info
                    with st.expander("Transcription Details"):
                        st.write(f"Language: {transcription.get('language', 'Unknown')}")
                        st.write(f"Duration: {transcription.get('duration', 0):.2f} seconds")
                        st.write(f"Processing time: {transcription.get('transcription_time', 0):.2f} seconds")
                        
                        if transcription.get("segments"):
                            st.write("**Segments:**")
                            for i, segment in enumerate(transcription["segments"][:5]):  # Show first 5 segments
                                st.write(f"{i+1}. [{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
                else:
                    st.error("No text was transcribed")
                    
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
    
    # Instructions
    st.subheader("How to use:")
    st.write("1. Upload an audio file (WAV, MP3, M4A, or FLAC)")
    st.write("2. The audio will be played back for preview")
    st.write("3. Click 'Transcribe Audio' to convert speech to text")
    st.write("4. View the transcription results and details")
    
    # Tips for better results
    st.subheader("Tips for better transcription:")
    st.write("- Speak clearly and at a normal pace")
    st.write("- Minimize background noise")
    st.write("- Keep the microphone close to your mouth")
    st.write("- Avoid speaking too quickly or too slowly")

if __name__ == "__main__":
    main() 