"""
RAG Engine module for podcast search system.

Orchestrates the complete RAG pipeline including retrieval and response generation.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from datetime import datetime

# Import our modules
from .audio_processor import AudioProcessor, process_podcast_episode
from .text_processor import TextProcessor, process_transcription_file
from .embeddings import EmbeddingManager

# Optional OpenAI for enhanced responses
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """Main RAG engine that orchestrates the complete pipeline."""
    
    def __init__(self, 
                 whisper_model: str = "base",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 collection_name: str = "podcast_episodes",
                 persist_directory: str = "./data/database",
                 openai_api_key: str = None):
        """
        Initialize the RAG engine.
        
        Args:
            whisper_model: Whisper model size
            embedding_model: Sentence transformer model name
            chunk_size: Text chunk size
            chunk_overlap: Chunk overlap size
            collection_name: ChromaDB collection name
            persist_directory: Database persistence directory
        """
        self.whisper_model = whisper_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize components
        self.audio_processor = AudioProcessor(model_name=whisper_model)
        self.text_processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Initialize OpenAI if available
        self.openai_client = None
        # Check for API key in order: parameter, environment variable, config file
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            self.openai_client = openai
            logger.info("OpenAI client initialized successfully")
        elif OPENAI_AVAILABLE:
            logger.info("OpenAI available but no API key provided")
        else:
            logger.info("OpenAI not available")
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Complete pipeline to process an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Starting complete processing pipeline for: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Step 1: Audio transcription
            logger.info("Step 1: Audio transcription")
            transcription_data = process_podcast_episode(audio_path)
            
            # Step 2: Text processing
            logger.info("Step 2: Text processing")
            transcription_path = f"data/transcripts/{transcription_data['episode_id']}.json"
            text_processing_result = process_transcription_file(transcription_path)
            
            # Step 3: Create embeddings
            logger.info("Step 3: Creating embeddings")
            chunks_path = text_processing_result["output_path"]
            embedding_result = self.embedding_manager.add_chunks_to_database(
                json.load(open(chunks_path, 'r', encoding='utf-8'))
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "episode_id": transcription_data["episode_id"],
                "audio_path": audio_path,
                "transcription_path": transcription_path,
                "chunks_path": chunks_path,
                "processing_time": processing_time,
                "transcription": {
                    "duration": transcription_data["metadata"]["duration"],
                    "segments": transcription_data["metadata"]["total_segments"],
                    "speakers": transcription_data["metadata"]["speakers"]
                },
                "text_processing": {
                    "chunks_created": text_processing_result["processed_chunks"],
                    "embedding_chunks": text_processing_result["embedding_chunks"]
                },
                "embeddings": {
                    "chunks_added": embedding_result["added"],
                    "errors": embedding_result["errors"]
                }
            }
            
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise
    
    def search_podcasts(self, 
                       query: str, 
                       n_results: int = 10,
                       filter_episode: str = None,
                       filter_speaker: str = None) -> Dict:
        """
        Search podcasts using the RAG system.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_episode: Optional episode ID filter
            filter_speaker: Optional speaker filter
            
        Returns:
            Dictionary with search results
        """
        logger.info(f"Searching podcasts for: '{query}'")
        
        start_time = time.time()
        
        # Build filter metadata
        filter_metadata = None
        if filter_episode:
            filter_metadata = {"episode_id": filter_episode}
        elif filter_speaker:
            filter_metadata = {"speaker": filter_speaker}
        
        # Search for similar chunks
        similar_chunks = self.embedding_manager.search_similar_chunks(
            query, n_results, filter_metadata
        )
        
        # Generate response if OpenAI is available
        response = None
        if self.openai_client and similar_chunks:
            response = self._generate_response(query, similar_chunks)
        
        search_time = time.time() - start_time
        
        result = {
            "query": query,
            "search_time": search_time,
            "total_results": len(similar_chunks),
            "results": similar_chunks,
            "generated_response": response,
            "filters": {
                "episode": filter_episode,
                "speaker": filter_speaker
            }
        }
        
        logger.info(f"Search completed in {search_time:.2f} seconds")
        return result
    
    def _generate_response(self, query: str, chunks: List[Dict]) -> str:
        """
        Generate a response using OpenAI based on retrieved chunks.
        
        Args:
            query: Original query
            chunks: Retrieved chunks
            
        Returns:
            Generated response
        """
        if not self.openai_client:
            return None
        
        try:
            # Prepare context from chunks
            context = "\n\n".join([
                f"Segment {i+1} (Episode: {chunk['metadata'].get('episode_id', 'Unknown')}, "
                f"Time: {chunk['metadata'].get('start_time', 0):.1f}s - {chunk['metadata'].get('end_time', 0):.1f}s, "
                f"Speaker: {chunk['metadata'].get('speaker', 'Unknown')}):\n{chunk['text']}"
                for i, chunk in enumerate(chunks)
            ])
            
            # Create prompt
            prompt = f"""Based on the following podcast segments, provide a comprehensive answer to the query.

Query: {query}

Podcast Segments:
{context}

Please provide a detailed answer that:
1. Directly addresses the query
2. References specific timestamps and speakers when relevant
3. Synthesizes information from multiple segments if applicable
4. Maintains the conversational tone of the original content

Answer:"""
            
            # Generate response
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based on podcast transcripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    def get_episode_summary(self, episode_id: str) -> Dict:
        """
        Get summary information for a specific episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            Dictionary with episode summary
        """
        return self.embedding_manager.get_episode_summary(episode_id)
    
    def get_database_stats(self) -> Dict:
        """
        Get overall database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        return self.embedding_manager.get_database_stats()
    
    def list_episodes(self) -> List[str]:
        """
        Get list of all episodes in the database.
        
        Returns:
            List of episode IDs
        """
        return self.embedding_manager.get_all_episodes()
    
    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode and all its data.
        
        Args:
            episode_id: Episode ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Delete from vector database
        success = self.embedding_manager.delete_episode(episode_id)
        
        if success:
            # Also delete transcription and chunk files
            try:
                transcription_path = f"data/transcripts/{episode_id}.json"
                chunks_path = f"data/processed/{episode_id}_chunks.json"
                
                if os.path.exists(transcription_path):
                    os.remove(transcription_path)
                    logger.info(f"Deleted transcription: {transcription_path}")
                
                if os.path.exists(chunks_path):
                    os.remove(chunks_path)
                    logger.info(f"Deleted chunks: {chunks_path}")
                    
            except Exception as e:
                logger.error(f"Error deleting files: {e}")
        
        return success
    
    def export_data(self, output_path: str) -> bool:
        """
        Export all data to a JSON file.
        
        Args:
            output_path: Path to save the export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get database stats
            stats = self.get_database_stats()
            
            # Get all episodes
            episodes = self.list_episodes()
            
            # Export embeddings
            embeddings_path = output_path.replace('.json', '_embeddings.json')
            self.embedding_manager.export_embeddings(embeddings_path)
            
            # Create export summary
            export_data = {
                "export_date": datetime.now().isoformat(),
                "system_info": {
                    "whisper_model": self.whisper_model,
                    "embedding_model": self.embedding_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                },
                "database_stats": stats,
                "episodes": episodes,
                "embeddings_file": embeddings_path
            }
            
            # Save export summary
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Data exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def batch_process_audio_files(self, audio_files: List[str]) -> List[Dict]:
        """
        Process multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file}")
            
            try:
                result = self.process_audio_file(audio_file)
                result["status"] = "success"
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                results.append({
                    "audio_path": audio_file,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def search_with_filters(self, 
                          query: str,
                          filters: Dict[str, Any]) -> Dict:
        """
        Advanced search with multiple filters.
        
        Args:
            query: Search query
            filters: Dictionary of filters
            
        Returns:
            Dictionary with search results
        """
        # Build filter metadata
        filter_metadata = {}
        
        if "episode_id" in filters:
            filter_metadata["episode_id"] = filters["episode_id"]
        
        if "speaker" in filters:
            filter_metadata["speaker"] = filters["speaker"]
        
        if "min_confidence" in filters:
            # Note: This would require additional implementation in ChromaDB
            pass
        
        # Perform search
        n_results = filters.get("n_results", 10)
        similar_chunks = self.embedding_manager.search_similar_chunks(
            query, n_results, filter_metadata if filter_metadata else None
        )
        
        # Apply additional filters if needed
        if "min_similarity" in filters:
            similar_chunks = [
                chunk for chunk in similar_chunks
                if chunk["similarity_score"] >= filters["min_similarity"]
            ]
        
        return {
            "query": query,
            "filters": filters,
            "results": similar_chunks,
            "total_results": len(similar_chunks)
        }


def create_rag_engine(config: Dict = None) -> RAGEngine:
    """
    Create a RAG engine with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RAGEngine instance
    """
    if config is None:
        config = {}
    
    return RAGEngine(
        whisper_model=config.get("whisper_model", "base"),
        embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 200),
        collection_name=config.get("collection_name", "podcast_episodes"),
        persist_directory=config.get("persist_directory", "./data/database"),
        openai_api_key=config.get("openai_api_key")
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # Initialize RAG engine
        rag_engine = create_rag_engine()
        
        if command == "process" and len(sys.argv) > 2:
            audio_file = sys.argv[2]
            result = rag_engine.process_audio_file(audio_file)
            print(f"Processed: {result['episode_id']}")
            
        elif command == "search" and len(sys.argv) > 2:
            query = sys.argv[2]
            results = rag_engine.search_podcasts(query)
            print(f"Found {results['total_results']} results")
            
        elif command == "stats":
            stats = rag_engine.get_database_stats()
            print(f"Database stats: {stats}")
            
        else:
            print("Usage:")
            print("  python rag_engine.py process <audio_file>")
            print("  python rag_engine.py search <query>")
            print("  python rag_engine.py stats")
    else:
        print("Usage: python rag_engine.py <command> [args]") 