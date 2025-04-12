import logging
import json
import requests
from typing import Dict, List, Any
from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, MAX_TOKENS

logger = logging.getLogger(__name__)

async def generate_rag_response(query: str, context: List[Dict[str, Any]]) -> str:
    """
    Generate a response using Retrieval-Augmented Generation (RAG) with Ollama.
    
    Args:
        query (str): The user's query
        context (List[Dict[str, Any]]): Context information retrieved from memory
        
    Returns:
        str: Generated response
    """
    try:
        # Format context for the prompt
        formatted_context = format_context_for_prompt(context)
        
        # Create the full prompt
        # Update the system prompt
        system_prompt = """You are an intelligent team memory assistant that helps teams recall information.
            You have access to team memory records which are provided to you in the user prompt.
            Your job is to answer questions based ONLY on the provided memory records.
            If the memory records contain the answer, provide it clearly and concisely.
            If the memory records don't contain enough information to answer the question, state that clearly.
            IMPORTANT: You DO have access to team memories - they will be provided in the prompt."""
            
        user_prompt = f"""
                Question: {query}

                Here are the relevant memory records:

                {formatted_context}

                Answer the question based ONLY on these memory records. If they don't contain enough information to answer the question, simply state that. If they do contain the answer, provide it clearly.
                """
        
        # Generate response using Ollama
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_LLM_MODEL,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": MAX_TOKENS
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json().get("response", "")
        
    except Exception as e:
        logger.error(f"Error in RAG generation: {e}", exc_info=True)
        return "I'm sorry, I couldn't generate a response based on the team memory."

def format_context_for_prompt(context: List[Dict[str, Any]]) -> str:
    """
    Format the context information for inclusion in the prompt.
    
    Args:
        context (List[Dict[str, Any]]): Context information
        
    Returns:
        str: Formatted context string
    """
    # Add debug logging
    logger.info(f"Formatting {len(context)} context items for prompt")
    
    formatted_context = ""
    
    for i, item in enumerate(context, 1):
        # Print the full content, not just a snippet
        if item.get("source") == "mem0":
            summary = item.get("summary", "No summary available")
            logger.info(f"Full Mem0 item {i} content: {summary}")

        
        if item.get("source") == "vector_store":
            # Format vector store item
            metadata = item.get("metadata", {})
            text = item.get("text", "No content available")
            
            # Log the content
            logger.info(f"Vector item {i} content: {text[:100]}...")
            
            formatted_context += f"MEMORY {i}:\n"
            formatted_context += f"Type: {metadata.get('type', 'unknown')}\n"
            formatted_context += f"Date: {metadata.get('timestamp', 'unknown')}\n"
            formatted_context += f"Content: {text}\n"
            if "participants" in metadata:
                formatted_context += f"Participants: {', '.join(metadata.get('participants', []))}\n"
            formatted_context += "\n"
            
        elif item.get("source") == "mem0":
            # Format Mem0 item
            # Log the content
            summary = item.get("summary", "No summary available")
            logger.info(f"Mem0 item {i} content: {summary[:100]}...")
            
            formatted_context += f"MEMORY {i}:\n"
            formatted_context += f"Type: {item.get('type', 'unknown')}\n"
            formatted_context += f"Date: {item.get('timestamp', 'unknown')}\n"
            formatted_context += f"Content: {summary}\n"
            formatted_context += f"Context: {item.get('context', 'No context available')}\n"
            if "participants" in item:
                formatted_context += f"Participants: {', '.join(item.get('participants', []))}\n"
            formatted_context += "\n"
    
    # Log the length of the formatted context
    logger.info(f"Formatted context length: {len(formatted_context)} characters")
    
    return formatted_context

def create_rag_prompt(query: str, context: str) -> str:
    """
    Create the full prompt for RAG.
    
    Args:
        query (str): The user's query
        context (str): Formatted context information
        
    Returns:
        str: Complete prompt for the LLM
    """
    return f"""
I need you to answer a question based on the team's memory records.

Question: {query}

Here are the relevant memory records from the team's history:

{context}

Please provide a clear, concise answer based ONLY on the information in these memory records.
If the records don't contain enough information to answer the question, please state that clearly.
Cite specific memories when appropriate by referring to their type and date.
"""