"""
Utility functions for podcast RAG system.

Helper functions for file handling, validation, and common operations.
"""

import os
import logging
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into a readable timestamp.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    Format duration in a human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate audio file format and properties.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size (max 100MB)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    if file_size > 100:
        return False, f"File too large: {file_size:.2f}MB (max 100MB)"
    
    # Check file extension
    valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        return False, f"Unsupported file format: {file_ext}"
    
    return True, ""


def get_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA-256 hash string
    """
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def create_episode_id(audio_path: str) -> str:
    """
    Create a unique episode ID from audio file path.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Unique episode ID
    """
    # Get file name without extension
    file_name = Path(audio_path).stem
    
    # Remove special characters and replace spaces with underscores
    episode_id = re.sub(r'[^\w\s-]', '', file_name)
    episode_id = re.sub(r'[-\s]+', '_', episode_id)
    episode_id = episode_id.lower()
    
    # Add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{episode_id}_{timestamp}"


def save_processing_log(log_data: Dict, log_file: str = "processing_log.json"):
    """
    Save processing log to JSON file.
    
    Args:
        log_data: Log data dictionary
        log_file: Path to log file
    """
    log_data["timestamp"] = datetime.now().isoformat()
    
    # Load existing logs
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading existing log: {e}")
    
    # Add new log entry
    logs.append(log_data)
    
    # Save updated logs
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        logger.info(f"Processing log saved to: {log_file}")
    except Exception as e:
        logger.error(f"Error saving processing log: {e}")


def load_processing_log(log_file: str = "processing_log.json") -> List[Dict]:
    """
    Load processing log from JSON file.
    
    Args:
        log_file: Path to log file
        
    Returns:
        List of log entries
    """
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        return logs
    except Exception as e:
        logger.error(f"Error loading processing log: {e}")
        return []


def get_audio_files_in_directory(directory: str) -> List[str]:
    """
    Get all audio files in a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        List of audio file paths
    """
    valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    audio_files = []
    
    if not os.path.exists(directory):
        return audio_files
    
    for file_path in Path(directory).rglob("*"):
        if file_path.suffix.lower() in valid_extensions:
            audio_files.append(str(file_path))
    
    return sorted(audio_files)


def create_directory_structure(base_dir: str = ".") -> Dict[str, str]:
    """
    Create the necessary directory structure for the project.
    
    Args:
        base_dir: Base directory for the project
        
    Returns:
        Dictionary of created directories
    """
    directories = {
        "audio": os.path.join(base_dir, "data", "audio"),
        "transcripts": os.path.join(base_dir, "data", "transcripts"),
        "processed": os.path.join(base_dir, "data", "processed"),
        "database": os.path.join(base_dir, "data", "database"),
        "logs": os.path.join(base_dir, "logs"),
        "exports": os.path.join(base_dir, "exports")
    }
    
    created_dirs = {}
    
    for name, path in directories.items():
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            created_dirs[name] = path
            logger.info(f"Created directory: {path}")
        except Exception as e:
            logger.error(f"Error creating directory {path}: {e}")
    
    return created_dirs


def clean_text_for_display(text: str, max_length: int = 200) -> str:
    """
    Clean and truncate text for display.
    
    Args:
        text: Raw text
        max_length: Maximum length for display
        
    Returns:
        Cleaned and truncated text
    """
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned


def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple heuristics.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Convert to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Filter out stop words and short words
    keywords = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]
    
    # Count frequency
    from collections import Counter
    keyword_counts = Counter(keywords)
    
    # Return top keywords
    return [kw for kw, _ in keyword_counts.most_common(max_keywords)]


def calculate_similarity_score(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using word overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lowercase and split into words
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def format_search_results(results: List[Dict]) -> List[Dict]:
    """
    Format search results for display.
    
    Args:
        results: Raw search results
        
    Returns:
        Formatted search results
    """
    formatted_results = []
    
    for i, result in enumerate(results, 1):
        formatted_result = {
            "rank": i,
            "text": clean_text_for_display(result["text"]),
            "full_text": result["text"],
            "similarity_score": f"{result['similarity_score']:.3f}",
            "timestamp": format_timestamp(result["metadata"].get("start_time", 0)),
            "duration": format_duration(
                result["metadata"].get("end_time", 0) - result["metadata"].get("start_time", 0)
            ),
            "speaker": result["metadata"].get("speaker", "Unknown"),
            "episode_id": result["metadata"].get("episode_id", "Unknown"),
            "word_count": result["metadata"].get("word_count", 0)
        }
        formatted_results.append(formatted_result)
    
    return formatted_results


def create_search_summary(search_results: Dict) -> Dict:
    """
    Create a summary of search results.
    
    Args:
        search_results: Search results dictionary
        
    Returns:
        Summary dictionary
    """
    if not search_results["results"]:
        return {
            "total_results": 0,
            "episodes_found": 0,
            "speakers_found": 0,
            "avg_similarity": 0.0,
            "total_duration": 0.0
        }
    
    results = search_results["results"]
    
    # Extract unique episodes and speakers
    episodes = set(r["metadata"].get("episode_id", "") for r in results)
    speakers = set(r["metadata"].get("speaker", "") for r in results)
    
    # Calculate statistics
    avg_similarity = sum(r["similarity_score"] for r in results) / len(results)
    total_duration = sum(
        r["metadata"].get("end_time", 0) - r["metadata"].get("start_time", 0)
        for r in results
    )
    
    return {
        "total_results": len(results),
        "episodes_found": len(episodes),
        "speakers_found": len(speakers),
        "avg_similarity": avg_similarity,
        "total_duration": total_duration,
        "episodes": list(episodes),
        "speakers": list(speakers)
    }


def validate_config(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ["whisper_model", "embedding_model"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate model names
    valid_whisper_models = {"tiny", "base", "small", "medium", "large"}
    if "whisper_model" in config and config["whisper_model"] not in valid_whisper_models:
        errors.append(f"Invalid Whisper model: {config['whisper_model']}")
    
    # Validate numeric fields
    numeric_fields = ["chunk_size", "chunk_overlap"]
    for field in numeric_fields:
        if field in config:
            try:
                value = int(config[field])
                if value <= 0:
                    errors.append(f"{field} must be positive")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a valid integer")
    
    return len(errors) == 0, errors


def load_config(config_file: str = "config.json") -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "whisper_model": "base",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "collection_name": "podcast_episodes",
        "persist_directory": "./data/database"
    }
    
    if not os.path.exists(config_file):
        logger.info(f"Config file not found, using defaults: {config_file}")
        return default_config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config
        is_valid, errors = validate_config(config)
        if not is_valid:
            logger.warning(f"Config validation errors: {errors}")
            logger.info("Using default configuration")
            return default_config
        
        # Merge with defaults
        merged_config = default_config.copy()
        merged_config.update(config)
        
        logger.info(f"Configuration loaded from: {config_file}")
        return merged_config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Using default configuration")
        return default_config


def save_config(config: Dict, config_file: str = "config.json"):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to configuration file
    """
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to: {config_file}")
    except Exception as e:
        logger.error(f"Error saving config: {e}")


if __name__ == "__main__":
    # Example usage
    print("Utility functions for podcast RAG system")
    print(f"Timestamp format: {format_timestamp(3661.5)}")
    print(f"Duration format: {format_duration(3661.5)}")
    
    # Create directory structure
    dirs = create_directory_structure()
    print(f"Created directories: {list(dirs.keys())}") 