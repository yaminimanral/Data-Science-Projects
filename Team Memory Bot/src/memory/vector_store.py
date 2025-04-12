import logging
import os
import json
import requests
from typing import List, Dict, Any, Optional
import asyncio
from config import (
    CHROMADB_PERSIST_DIRECTORY, OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL
)

logger = logging.getLogger(__name__)

# Global variable to hold the vector store client
_vector_client = None

async def initialize_vector_db():
    """Initialize the ChromaDB vector database."""
    await initialize_chromadb()

async def initialize_chromadb():
    """Initialize ChromaDB vector database."""
    persist_dir = CHROMADB_PERSIST_DIRECTORY
    abs_path = os.path.abspath(persist_dir)
    logger.info(f"ChromaDB directory being used: {abs_path}")
    logger.info(f"Directory exists: {os.path.exists(abs_path)}")
    try:
        # Import ChromaDB
        import chromadb
        from chromadb.config import Settings
        
        # Ensure the persist directory exists
        os.makedirs(CHROMADB_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize client with persistence
        client = chromadb.PersistentClient(path=CHROMADB_PERSIST_DIRECTORY)
        
        # List all collections first (for debugging)
        all_collections = client.list_collections()
        logger.info(f"Existing collections: {[c.name for c in all_collections]}")
        
        # Create or get collection for storing embeddings
        collection_name = "team_memory"
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Connected to existing ChromaDB collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist, create it (catch any exception, not just ValueError)
            logger.info(f"Collection '{collection_name}' not found, creating it. Error was: {str(e)}")
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new ChromaDB collection: {collection_name}")
        
        global _vector_client
        _vector_client = collection
        
        logger.info("ChromaDB initialized successfully")
        
    except ImportError:
        logger.error("ChromaDB package not installed. Please install it with: pip install chromadb")
        raise
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}", exc_info=True)
        raise

async def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using Ollama.
    
    Args:
        text (str): Text to embed
        
    Returns:
        List[float]: Embedding vector
    """
    try:
        # Using Ollama for embeddings
        url = f"{OLLAMA_BASE_URL}/api/embeddings"
        payload = {
            "model": OLLAMA_EMBEDDING_MODEL,
            "prompt": text
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        embedding = response.json().get("embedding", [])
        
        if not embedding:
            logger.error("Empty embedding returned from Ollama")
            # Return a zero vector as fallback
            return [0.0] * 384  # Most Ollama embedding models use 384 dimensions
            
        return embedding
        
    except ImportError:
        logger.error("Requests package not installed. Please install it with: pip install requests")
        raise
    except Exception as e:
        logger.error(f"Error generating embedding with Ollama: {e}", exc_info=True)
        # Return a zero vector as fallback (not ideal but prevents system crash)
        return [0.0] * 384  # Most Ollama embedding models use 384 dimensions

async def store_vector(memory_id: str, text: str, metadata: Dict[str, Any]):
    """
    Store a vector embedding in the vector database.
    
    Args:
        memory_id (str): Unique ID for the memory
        text (str): Text to embed and store
        metadata (Dict[str, Any]): Additional metadata to store with the vector
    """
    if _vector_client is None:
        await initialize_vector_db()
    
    try:
        # Get embedding for the text
        embedding = await get_embedding(text)
        
        # Store in ChromaDB
        _vector_client.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )
        
        logger.debug(f"Stored vector for memory_id: {memory_id}")
        
    except Exception as e:
        logger.error(f"Error storing vector: {e}", exc_info=True)

async def query_vectors(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query vectors by semantic similarity.
    
    Args:
        query_text (str): Text to search for
        top_k (int): Number of results to return
        
    Returns:
        List[Dict[str, Any]]: List of matching memories with their metadata
    """
    if _vector_client is None:
        await initialize_vector_db()
    
    try:
        # Get embedding for the query
        embedding = await get_embedding(query_text)
        
        # Query ChromaDB
        results = _vector_client.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        
        # Format results
        matches = []
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            matches.append({
                "id": doc_id,
                "score": 1.0 - distance,  # Convert distance to similarity score
                "metadata": metadata,
                "text": document
            })
        
        return matches
        
    except Exception as e:
        logger.error(f"Error querying vectors: {e}", exc_info=True)
        return []