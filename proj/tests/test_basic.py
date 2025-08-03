"""
Basic tests for the podcast RAG system.
"""

import unittest
import tempfile
import os
from pathlib import Path
import json

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import (
    format_timestamp, format_duration, validate_audio_file,
    create_directory_structure, clean_text_for_display
)


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests."""
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        self.assertEqual(format_timestamp(61.5), "01:01")
        self.assertEqual(format_timestamp(3661.5), "01:01:01")
        self.assertEqual(format_timestamp(0), "00:00")
    
    def test_format_duration(self):
        """Test duration formatting."""
        self.assertEqual(format_duration(30), "30.0s")
        self.assertEqual(format_duration(90), "1.5m")
        self.assertEqual(format_duration(7200), "2.0h")
    
    def test_clean_text_for_display(self):
        """Test text cleaning for display."""
        text = "This is a very long text that should be truncated for display purposes"
        cleaned = clean_text_for_display(text, max_length=20)
        self.assertLessEqual(len(cleaned), 23)  # 20 + "..."
        self.assertTrue(cleaned.endswith("..."))
    
    def test_create_directory_structure(self):
        """Test directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dirs = create_directory_structure(temp_dir)
            self.assertIn("audio", dirs)
            self.assertIn("transcripts", dirs)
            self.assertIn("database", dirs)
            
            # Check if directories were actually created
            for name, path in dirs.items():
                self.assertTrue(os.path.exists(path))
    
    def test_validate_audio_file(self):
        """Test audio file validation."""
        # Test with non-existent file
        is_valid, error_msg = validate_audio_file("nonexistent.mp3")
        self.assertFalse(is_valid)
        self.assertIn("does not exist", error_msg)
        
        # Test with valid extension
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"dummy content")
            temp_file = f.name
        
        try:
            is_valid, error_msg = validate_audio_file(temp_file)
            self.assertTrue(is_valid)
            self.assertEqual(error_msg, "")
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main() 