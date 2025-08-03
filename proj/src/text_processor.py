"""
Text processing module for podcast RAG system.

Handles text chunking, cleaning, and preparation for vector embeddings.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text processing, chunking, and cleaning for RAG system."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load spaCy model for better text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Get stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Common podcast filler words to remove
        self.filler_words = {
            'um', 'uh', 'ah', 'er', 'hmm', 'like', 'you know', 'i mean',
            'basically', 'actually', 'literally', 'sort of', 'kind of'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove filler words
        for filler in self.filler_words:
            text = text.replace(filler, '')
        
        # Clean up multiple punctuation
        text = re.sub(r'[\.\!\?]+', '.', text)
        text = re.sub(r'[\,\;]+', ',', text)
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def create_semantic_chunks(self, text: str) -> List[Dict]:
        """
        Create semantic chunks from text with metadata.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks with metadata
        """
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = self.split_into_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_data = {
                    "text": current_chunk.strip(),
                    "start_sentence": chunk_start,
                    "end_sentence": i - 1,
                    "char_start": 0,  # Will be updated with actual positions
                    "char_end": len(current_chunk),
                    "word_count": len(current_chunk.split()),
                    "sentence_count": i - chunk_start
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + " " + sentence
                chunk_start = i - len(overlap_text.split()) if overlap_text else i
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunk_data = {
                "text": current_chunk.strip(),
                "start_sentence": chunk_start,
                "end_sentence": len(sentences) - 1,
                "char_start": 0,
                "char_end": len(current_chunk),
                "word_count": len(current_chunk.split()),
                "sentence_count": len(sentences) - chunk_start
            }
            chunks.append(chunk_data)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def process_episode_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Process episode segments into searchable chunks.
        
        Args:
            segments: List of episode segments with timestamps
            
        Returns:
            List of processed chunks with metadata
        """
        processed_chunks = []
        
        for segment in segments:
            # Clean segment text
            cleaned_text = self.clean_text(segment["text"])
            
            # Skip empty or very short segments
            if len(cleaned_text) < 10:
                continue
            
            # Create chunk data
            chunk_data = {
                "text": cleaned_text,
                "start_time": segment["start"],
                "end_time": segment["end"],
                "speaker": segment.get("speaker", "Unknown"),
                "confidence": segment.get("confidence", 0),
                "episode_id": segment.get("episode_id", ""),
                "segment_id": segment.get("segment_id", ""),
                "word_count": len(cleaned_text.split()),
                "duration": segment["end"] - segment["start"]
            }
            
            processed_chunks.append(chunk_data)
        
        logger.info(f"Processed {len(processed_chunks)} segments into chunks")
        return processed_chunks
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text using spaCy.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        doc = self.nlp(text)
        
        # Extract nouns, verbs, and adjectives
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Count frequency and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(max_keywords)]
        
        return top_keywords
    
    def create_chunk_embeddings_ready(self, chunks: List[Dict]) -> List[Dict]:
        """
        Prepare chunks for embedding generation.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks ready for embedding
        """
        embedding_ready_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Extract keywords
            keywords = self.extract_keywords(chunk["text"])
            
            # Create embedding-ready chunk
            embedding_chunk = {
                "id": f"chunk_{i}",
                "text": chunk["text"],
                "metadata": {
                    "keywords": keywords,
                    "word_count": chunk.get("word_count", 0),
                    "sentence_count": chunk.get("sentence_count", 0),
                    "start_time": chunk.get("start_time", 0),
                    "end_time": chunk.get("end_time", 0),
                    "speaker": chunk.get("speaker", "Unknown"),
                    "episode_id": chunk.get("episode_id", ""),
                    "confidence": chunk.get("confidence", 0)
                }
            }
            
            embedding_ready_chunks.append(embedding_chunk)
        
        logger.info(f"Prepared {len(embedding_ready_chunks)} chunks for embedding")
        return embedding_ready_chunks
    
    def merge_similar_chunks(self, chunks: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Merge similar chunks to reduce redundancy.
        
        Args:
            chunks: List of chunks to merge
            similarity_threshold: Threshold for similarity
            
        Returns:
            List of merged chunks
        """
        # Simple implementation - in production, use proper similarity metrics
        merged_chunks = []
        used_indices = set()
        
        for i, chunk1 in enumerate(chunks):
            if i in used_indices:
                continue
                
            similar_chunks = [chunk1]
            used_indices.add(i)
            
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Simple text similarity (can be improved with embeddings)
                similarity = self._calculate_text_similarity(chunk1["text"], chunk2["text"])
                
                if similarity > similarity_threshold:
                    similar_chunks.append(chunk2)
                    used_indices.add(j)
            
            # Merge similar chunks
            if len(similar_chunks) > 1:
                merged_text = " ".join([c["text"] for c in similar_chunks])
                merged_chunk = {
                    "text": merged_text,
                    "metadata": {
                        "merged_from": len(similar_chunks),
                        "original_chunks": [c.get("id", "") for c in similar_chunks]
                    }
                }
                merged_chunks.append(merged_chunk)
            else:
                merged_chunks.append(chunk1)
        
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity using word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


def process_transcription_file(transcription_path: str, output_dir: str = "data/processed") -> Dict:
    """
    Process a transcription file into searchable chunks.
    
    Args:
        transcription_path: Path to transcription JSON file
        output_dir: Directory to save processed chunks
        
    Returns:
        Dictionary with processing results
    """
    processor = TextProcessor()
    
    # Load transcription
    with open(transcription_path, 'r', encoding='utf-8') as f:
        transcription_data = json.load(f)
    
    # Process segments
    segments = transcription_data.get("segments", [])
    processed_chunks = processor.process_episode_segments(segments)
    
    # Prepare for embeddings
    embedding_chunks = processor.create_chunk_embeddings_ready(processed_chunks)
    
    # Save processed chunks
    output_path = Path(output_dir) / f"{Path(transcription_path).stem}_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embedding_chunks, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processed transcription saved to: {output_path}")
    
    return {
        "episode_id": transcription_data.get("episode_id", ""),
        "total_segments": len(segments),
        "processed_chunks": len(processed_chunks),
        "embedding_chunks": len(embedding_chunks),
        "output_path": str(output_path)
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        transcription_file = sys.argv[1]
        result = process_transcription_file(transcription_file)
        print(f"Processed episode: {result['episode_id']}")
        print(f"Segments: {result['total_segments']}")
        print(f"Chunks: {result['processed_chunks']}")
    else:
        print("Usage: python text_processor.py <transcription_file>") 