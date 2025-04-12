import logging
import requests
from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL

logger = logging.getLogger(__name__)

async def extract_knowledge(text: str) -> str:
    """
    Extract key knowledge from a text message using Ollama LLM.
    
    Args:
        text (str): The raw text message to analyze
        
    Returns:
        str: A concise summary of the extracted knowledge
    """
    if not text or len(text.strip()) < 10:
        logger.debug("Text too short for knowledge extraction")
        return ""
    
    try:
        system_prompt = "You are a precise and accurate knowledge extraction assistant."
        
        user_prompt = f"""
        You are a knowledge extraction AI that identifies important team information from messages. 
        Extract the key knowledge, decisions, or blockers from the following message.
        If there's nothing important, respond with "No significant knowledge".
        
        Provide a concise summary in 1-2 sentences that captures the essence.
        
        Message: {text}
        
        Extracted knowledge:
        """
        
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_LLM_MODEL,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": 150,
                "temperature": 0.1
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        extraction = response.json().get("response", "").strip()
        
        # If the model determines there's no significant knowledge, return empty string
        if "No significant knowledge" in extraction:
            return ""
            
        return extraction
        
    except Exception as e:
        logger.error(f"Error in knowledge extraction: {e}", exc_info=True)
        return ""