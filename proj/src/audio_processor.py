"""
Audio processing module for podcast RAG system.

Handles audio preprocessing, speech-to-text conversion, and speaker diarization.
"""

import os
import logging
import tempfile
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time

import whisper
import torch
import torchaudio
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio processing, transcription, and speaker diarization."""
    
    def __init__(self, model_name: str = "base", device: str = None):
        """
        Initialize the audio processor.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run models on (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_name}")
        self.whisper_model = whisper.load_model(model_name, device=self.device)
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.channels = 1
        
    def preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file for optimal transcription.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file
        """
        logger.info(f"Preprocessing audio: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted stereo to mono")
        
        # Resample to target sample rate
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
            logger.info(f"Resampled to {self.sample_rate}Hz")
        
        # Normalize audio
        audio = audio.normalize()
        
        # Create temporary file for preprocessed audio
        temp_dir = Path("data/transcripts")
        temp_dir.mkdir(exist_ok=True)
        
        preprocessed_path = temp_dir / f"preprocessed_{Path(audio_path).stem}.wav"
        audio.export(str(preprocessed_path), format="wav")
        
        logger.info(f"Preprocessed audio saved to: {preprocessed_path}")
        return str(preprocessed_path)
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio using Whisper with timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            
        Returns:
            Dictionary containing transcription data
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        # Preprocess audio
        preprocessed_path = self.preprocess_audio(audio_path)
        
        # Transcribe with timestamps
        start_time = time.time()
        
        if language:
            result = self.whisper_model.transcribe(
                preprocessed_path,
                language=language,
                word_timestamps=True,
                verbose=True
            )
        else:
            result = self.whisper_model.transcribe(
                preprocessed_path,
                word_timestamps=True,
                verbose=True
            )
        
        transcription_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
        
        # Process results
        transcription_data = {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
            "duration": result.get("duration", 0),
            "transcription_time": transcription_time,
            "audio_path": audio_path,
            "preprocessed_path": preprocessed_path
        }
        
        return transcription_data
    
    def extract_segments_with_timestamps(self, transcription_data: Dict) -> List[Dict]:
        """
        Extract text segments with precise timestamps.
        
        Args:
            transcription_data: Output from transcribe_audio
            
        Returns:
            List of segments with timestamps and text
        """
        segments = []
        
        for segment in transcription_data["segments"]:
            segment_data = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "confidence": segment.get("avg_logprob", 0),
                "words": segment.get("words", [])
            }
            segments.append(segment_data)
        
        logger.info(f"Extracted {len(segments)} segments")
        return segments
    
    def detect_speakers(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """
        Detect and label speakers in audio segments.
        
        Args:
            audio_path: Path to audio file
            segments: List of transcription segments
            
        Returns:
            List of segments with speaker labels
        """
        logger.info("Speaker detection not implemented in this version")
        logger.info("Using placeholder speaker labels")
        
        # Placeholder speaker detection
        # In a full implementation, this would use pyannote.audio or similar
        labeled_segments = []
        
        for i, segment in enumerate(segments):
            # Simple heuristic: alternate speakers for demonstration
            speaker_id = f"Speaker_{i % 2 + 1}"
            
            labeled_segment = segment.copy()
            labeled_segment["speaker"] = speaker_id
            labeled_segments.append(labeled_segment)
        
        return labeled_segments
    
    def save_transcription(self, transcription_data: Dict, output_path: str):
        """
        Save transcription data to JSON file.
        
        Args:
            transcription_data: Transcription data dictionary
            output_path: Path to save the JSON file
        """
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcription saved to: {output_path}")
    
    def load_transcription(self, transcription_path: str) -> Dict:
        """
        Load transcription data from JSON file.
        
        Args:
            transcription_path: Path to transcription JSON file
            
        Returns:
            Transcription data dictionary
        """
        with open(transcription_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        logger.info(f"Transcription loaded from: {transcription_path}")
        return transcription_data
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate audio file format and size.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Check file size (max 100MB)
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
            if file_size > 100:
                logger.error(f"Audio file too large: {file_size:.2f}MB")
                return False
            
            # Check duration (max 2 hours)
            duration = self.get_audio_duration(audio_path)
            if duration > 7200:  # 2 hours
                logger.error(f"Audio file too long: {duration:.2f} seconds")
                return False
            
            # Try to load audio
            audio = AudioSegment.from_file(audio_path)
            logger.info(f"Audio file validated: {duration:.2f}s, {file_size:.2f}MB")
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            return False


def process_podcast_episode(audio_path: str, output_dir: str = "data/transcripts") -> Dict:
    """
    Complete pipeline to process a podcast episode.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with processing results
    """
    processor = AudioProcessor()
    
    # Validate audio file
    if not processor.validate_audio_file(audio_path):
        raise ValueError(f"Invalid audio file: {audio_path}")
    
    # Transcribe audio
    transcription_data = processor.transcribe_audio(audio_path)
    
    # Extract segments
    segments = processor.extract_segments_with_timestamps(transcription_data)
    
    # Detect speakers
    labeled_segments = processor.detect_speakers(audio_path, segments)
    
    # Prepare output data
    episode_data = {
        "episode_id": Path(audio_path).stem,
        "audio_path": audio_path,
        "transcription": transcription_data["text"],
        "segments": labeled_segments,
        "metadata": {
            "language": transcription_data["language"],
            "duration": transcription_data["duration"],
            "transcription_time": transcription_data["transcription_time"],
            "total_segments": len(labeled_segments),
            "speakers": list(set(seg["speaker"] for seg in labeled_segments))
        }
    }
    
    # Save transcription
    output_path = Path(output_dir) / f"{episode_data['episode_id']}.json"
    processor.save_transcription(episode_data, str(output_path))
    
    logger.info(f"Episode processing completed: {episode_data['episode_id']}")
    return episode_data


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        result = process_podcast_episode(audio_file)
        print(f"Processed episode: {result['episode_id']}")
        print(f"Duration: {result['metadata']['duration']:.2f}s")
        print(f"Segments: {result['metadata']['total_segments']}")
    else:
        print("Usage: python audio_processor.py <audio_file>") 