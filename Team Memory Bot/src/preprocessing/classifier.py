import logging
import requests
from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, MEMORY_TYPES

logger = logging.getLogger(__name__)

async def classify_message(text: str) -> str:
    """
    Classify a message into a memory type (decision, blocker, etc.) using Ollama.
    
    Args:
        text (str): The message text to classify
        
    Returns:
        str: The classified memory type
    """
    if not text or len(text.strip()) < 10:
        return "status_update"  # Default classification for short messages
    
    try:
        # Format memory types for prompt
        memory_types_str = ", ".join(MEMORY_TYPES)
        
        system_prompt = "You are a precise message classifier that responds with a single word classification."
        
        user_prompt = f"""
        Classify the following message into one of these types: {memory_types_str}.
        
        Here are guidelines for each type:
        - decision: Contains a conclusion or choice made by the team
        - blocker: Identifies an obstacle or impediment to progress
        - status_update: Provides information about current progress
        - milestone: Marks completion of a significant project phase
        - question: Asks for information or clarification
        - answer: Provides information in response to a question
        
        Message: {text}
        
        Classification (single word only):
        """
        
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_LLM_MODEL,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0.1
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Extract and clean classification
        classification = response.json().get("response", "").strip().lower()
        
        # Validate against known types
        if classification not in MEMORY_TYPES:
            logger.warning(f"Unknown classification returned: {classification}, defaulting to status_update")
            return "status_update"
            
        return classification
        
    except Exception as e:
        logger.error(f"Error in message classification: {e}", exc_info=True)
        return "status_update"  # Default classification on error