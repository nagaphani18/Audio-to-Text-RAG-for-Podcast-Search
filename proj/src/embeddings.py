"""
Embeddings module for podcast RAG system.

Handles vector embeddings generation and similarity search using sentence transformers.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages vector embeddings and similarity search for the RAG system."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 collection_name: str = "podcast_episodes",
                 persist_directory: str = "./data/database"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Sentence transformer model name
            collection_name: ChromaDB collection name
            persist_directory: Directory to persist ChromaDB
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Podcast episode embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def add_chunks_to_database(self, chunks: List[Dict]) -> Dict:
        """
        Add text chunks to the vector database.
        
        Args:
            chunks: List of chunks with text and metadata
            
        Returns:
            Dictionary with operation results
        """
        if not chunks:
            logger.warning("No chunks provided")
            return {"added": 0, "errors": 0}
        
        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add to ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to database")
            return {"added": len(chunks), "errors": 0}
            
        except Exception as e:
            logger.error(f"Error adding chunks to database: {e}")
            return {"added": 0, "errors": len(chunks)}
    
    def search_similar_chunks(self, 
                            query: str, 
                            n_results: int = 10,
                            filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar chunks with scores
        """
        logger.info(f"Searching for: '{query}' (n_results={n_results})")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            similar_chunks = []
            for i in range(len(results["ids"][0])):
                chunk_data = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                }
                similar_chunks.append(chunk_data)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []
    
    def search_by_episode(self, 
                         query: str, 
                         episode_id: str,
                         n_results: int = 10) -> List[Dict]:
        """
        Search for similar chunks within a specific episode.
        
        Args:
            query: Search query
            episode_id: Episode ID to search in
            n_results: Number of results to return
            
        Returns:
            List of similar chunks from the specified episode
        """
        filter_metadata = {"episode_id": episode_id}
        return self.search_similar_chunks(query, n_results, filter_metadata)
    
    def search_by_speaker(self, 
                         query: str, 
                         speaker: str,
                         n_results: int = 10) -> List[Dict]:
        """
        Search for similar chunks from a specific speaker.
        
        Args:
            query: Search query
            speaker: Speaker name to filter by
            n_results: Number of results to return
            
        Returns:
            List of similar chunks from the specified speaker
        """
        filter_metadata = {"speaker": speaker}
        return self.search_similar_chunks(query, n_results, filter_metadata)
    
    def get_episode_summary(self, episode_id: str) -> Dict:
        """
        Get summary statistics for an episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            Dictionary with episode statistics
        """
        try:
            # Get all chunks for the episode
            results = self.collection.get(
                where={"episode_id": episode_id}
            )
            
            if not results["ids"]:
                return {"episode_id": episode_id, "chunks": 0}
            
            # Calculate statistics
            total_chunks = len(results["ids"])
            total_words = sum(
                metadata.get("word_count", 0) 
                for metadata in results["metadatas"]
            )
            
            # Get unique speakers
            speakers = set(
                metadata.get("speaker", "Unknown") 
                for metadata in results["metadatas"]
            )
            
            # Calculate total duration
            total_duration = sum(
                metadata.get("end_time", 0) - metadata.get("start_time", 0)
                for metadata in results["metadatas"]
            )
            
            return {
                "episode_id": episode_id,
                "total_chunks": total_chunks,
                "total_words": total_words,
                "total_duration": total_duration,
                "speakers": list(speakers),
                "avg_chunk_length": total_words / total_chunks if total_chunks > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting episode summary: {e}")
            return {"episode_id": episode_id, "error": str(e)}
    
    def get_all_episodes(self) -> List[str]:
        """
        Get list of all episode IDs in the database.
        
        Returns:
            List of episode IDs
        """
        try:
            # Get all documents
            results = self.collection.get()
            
            # Extract unique episode IDs
            episode_ids = set(
                metadata.get("episode_id", "") 
                for metadata in results["metadatas"]
                if metadata.get("episode_id")
            )
            
            return list(episode_ids)
            
        except Exception as e:
            logger.error(f"Error getting episode list: {e}")
            return []
    
    def delete_episode(self, episode_id: str) -> bool:
        """
        Delete all chunks for a specific episode.
        
        Args:
            episode_id: Episode ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get chunks for the episode
            results = self.collection.get(
                where={"episode_id": episode_id}
            )
            
            if not results["ids"]:
                logger.warning(f"No chunks found for episode: {episode_id}")
                return True
            
            # Delete chunks
            self.collection.delete(ids=results["ids"])
            
            logger.info(f"Deleted {len(results['ids'])} chunks for episode: {episode_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting episode: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """
        Get overall database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get all documents
            results = self.collection.get()
            
            if not results["ids"]:
                return {"total_chunks": 0, "total_episodes": 0}
            
            total_chunks = len(results["ids"])
            
            # Count unique episodes
            episode_ids = set(
                metadata.get("episode_id", "") 
                for metadata in results["metadatas"]
                if metadata.get("episode_id")
            )
            
            # Count unique speakers
            speakers = set(
                metadata.get("speaker", "Unknown") 
                for metadata in results["metadatas"]
            )
            
            # Calculate total words and duration
            total_words = sum(
                metadata.get("word_count", 0) 
                for metadata in results["metadatas"]
            )
            
            total_duration = sum(
                metadata.get("end_time", 0) - metadata.get("start_time", 0)
                for metadata in results["metadatas"]
            )
            
            return {
                "total_chunks": total_chunks,
                "total_episodes": len(episode_ids),
                "total_speakers": len(speakers),
                "total_words": total_words,
                "total_duration": total_duration,
                "avg_chunk_length": total_words / total_chunks if total_chunks > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    def export_embeddings(self, output_path: str) -> bool:
        """
        Export embeddings and metadata to JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents
            results = self.collection.get()
            
            export_data = {
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "total_chunks": len(results["ids"]),
                "chunks": []
            }
            
            for i in range(len(results["ids"])):
                chunk_data = {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                export_data["chunks"].append(chunk_data)
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported embeddings to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting embeddings: {e}")
            return False


def create_embeddings_from_chunks(chunks_file: str, 
                                 output_dir: str = "data/database") -> Dict:
    """
    Create embeddings from processed chunks file.
    
    Args:
        chunks_file: Path to chunks JSON file
        output_dir: Directory to save embeddings
        
    Returns:
        Dictionary with processing results
    """
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager(persist_directory=output_dir)
    
    # Add chunks to database
    result = embedding_manager.add_chunks_to_database(chunks_data)
    
    # Get database stats
    stats = embedding_manager.get_database_stats()
    
    return {
        "chunks_file": chunks_file,
        "chunks_processed": len(chunks_data),
        "chunks_added": result["added"],
        "errors": result["errors"],
        "database_stats": stats
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        chunks_file = sys.argv[1]
        result = create_embeddings_from_chunks(chunks_file)
        print(f"Processed chunks file: {result['chunks_file']}")
        print(f"Chunks added: {result['chunks_added']}")
        print(f"Database stats: {result['database_stats']}")
    else:
        print("Usage: python embeddings.py <chunks_file>") 