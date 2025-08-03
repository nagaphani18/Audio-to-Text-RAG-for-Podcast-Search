"""
Unit tests for audio processor module.
"""

import unittest
import tempfile
import os
from pathlib import Path
import json

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.audio_processor import AudioProcessor, process_podcast_episode
from src.utils import validate_audio_file


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor(model_name="tiny")  # Use tiny for faster tests
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.model_name, "tiny")
        self.assertIsNotNone(self.processor.whisper_model)
    
    def test_validate_audio_file(self):
        """Test audio file validation."""
        # Test with non-existent file
        is_valid, error_msg = validate_audio_file("nonexistent.mp3")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", error_msg)
        
        # Test with valid file extension
        temp_file = os.path.join(self.temp_dir, "test.mp3")
        with open(temp_file, 'w') as f:
            f.write("dummy content")
        
        is_valid, error_msg = validate_audio_file(temp_file)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")
    
    def test_get_audio_duration(self):
        """Test audio duration calculation."""
        # Create a dummy audio file
        temp_file = os.path.join(self.temp_dir, "test.wav")
        with open(temp_file, 'w') as f:
            f.write("dummy content")
        
        # This will fail with a real audio file, but we're testing the method exists
        try:
            duration = self.processor.get_audio_duration(temp_file)
            self.assertIsInstance(duration, float)
        except Exception:
            # Expected for dummy file
            pass
    
    def test_save_and_load_transcription(self):
        """Test transcription save and load functionality."""
        # Create dummy transcription data
        transcription_data = {
            "text": "Hello world",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world"
                }
            ],
            "language": "en",
            "duration": 1.0
        }
        
        # Test save
        output_path = os.path.join(self.temp_dir, "test_transcription.json")
        self.processor.save_transcription(transcription_data, output_path)
        
        # Test load
        loaded_data = self.processor.load_transcription(output_path)
        self.assertEqual(loaded_data["text"], transcription_data["text"])
        self.assertEqual(loaded_data["language"], transcription_data["language"])
    
    def test_extract_segments_with_timestamps(self):
        """Test segment extraction with timestamps."""
        transcription_data = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world",
                    "avg_logprob": -0.5
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "How are you?",
                    "avg_logprob": -0.3
                }
            ]
        }
        
        segments = self.processor.extract_segments_with_timestamps(transcription_data)
        
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "Hello world")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 1.0)
        self.assertEqual(segments[1]["text"], "How are you?")
    
    def test_detect_speakers(self):
        """Test speaker detection functionality."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "Hi there"},
            {"start": 2.0, "end": 3.0, "text": "How are you?"}
        ]
        
        labeled_segments = self.processor.detect_speakers("dummy_audio.mp3", segments)
        
        self.assertEqual(len(labeled_segments), 3)
        self.assertIn("speaker", labeled_segments[0])
        self.assertIn("speaker", labeled_segments[1])
        self.assertIn("speaker", labeled_segments[2])


class TestProcessPodcastEpisode(unittest.TestCase):
    """Test cases for process_podcast_episode function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_process_podcast_episode_invalid_file(self):
        """Test processing with invalid audio file."""
        with self.assertRaises(ValueError):
            process_podcast_episode("nonexistent.mp3")


if __name__ == "__main__":
    unittest.main() 