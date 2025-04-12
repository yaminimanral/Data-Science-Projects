import logging
import json
import requests
from typing import Dict, List, Any, Optional
from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, TOP_K_RESULTS
from memory.mem0 import query_memories
from memory.vector_store import query_vectors
from query_engine.rag import generate_rag_response

logger = logging.getLogger(__name__)

async def query_knowledge(query: str, channel_id: Optional[str] = None) -> str:
    try:
        logger.info(f"Processing query: {query}")
        
        # Query both the vector store and Mem0 memory in parallel
        vector_results = await query_vectors(query, top_k=TOP_K_RESULTS)
        logger.info(f"Retrieved {len(vector_results)} results from vector store")
        
        # Use the channel_id if provided
        mem0_results = await query_memories(query, user_id=channel_id)
        
        # Log mem0 results
        mem0_count = len(mem0_results.get('memories', []))
        logger.info(f"Retrieved {mem0_count} memories from Mem0")
        
        # Print the first few Mem0 results for debugging
        for i, memory in enumerate(mem0_results.get('memories', [])[:3]):
            summary = memory.get('summary', 'No summary')
            logger.info(f"Mem0 memory {i+1}: {summary[:100]}...")
        
        # Combine and prepare context for RAG
        combined_context = prepare_context(vector_results, mem0_results)
        logger.info(f"Combined context has {len(combined_context)} items")
        
        # Generate response using RAG
        response = await generate_rag_response(query, combined_context)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in query_knowledge: {e}", exc_info=True)
        return f"I encountered an error while searching the team memory: {str(e)}"

def prepare_context(vector_results: List[Dict[str, Any]], mem0_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare and combine context from vector store and Mem0 results.
    """
    # Log what we're working with
    logger.info(f"Preparing context from {len(vector_results)} vector results and {len(mem0_results.get('memories', []))} Mem0 results")
    
    combined_context = []
    
    # Add vector results to context
    for result in vector_results:
        combined_context.append({
            "source": "vector_store",
            "id": result.get("id", "unknown"),
            "score": result.get("score", 0.0),
            "metadata": result.get("metadata", {}),
            "text": result.get("text", "")
        })
    
    # Add Mem0 results to context
    for memory in mem0_results.get("memories", []):
        # Log each memory being added
        summary = memory.get('summary', 'No summary')
        logger.info(f"Adding Mem0 memory to context: {summary[:100]}...")
        
        combined_context.append({
            "source": "mem0",
            "id": memory.get("id", "unknown"),
            "type": memory.get("type", "memory"),
            "summary": memory.get("summary", ""),
            "timestamp": memory.get("timestamp", ""),
            "context": memory.get("context", ""),
            "participants": memory.get("participants", [])
        })
    
    # Sort by relevance score if available
    combined_context.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    
    # Log the final count
    logger.info(f"Final combined context has {len(combined_context)} items")
    
    return combined_context