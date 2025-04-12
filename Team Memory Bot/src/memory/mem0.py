import logging
from typing import Dict, List, Any, Optional
import asyncio
import uuid
from config import MEM0_API_KEY

logger = logging.getLogger(__name__)

# Import the Mem0 client
try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    logger.error("mem0ai package not installed. Please install with: pip install mem0ai")
    MEM0_AVAILABLE = False

class Mem0Wrapper:
    """Wrapper for the official Mem0 MemoryClient that provides async functionality."""
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        if not MEM0_AVAILABLE:
            logger.error("mem0ai package not installed. Cannot initialize client.")
            return
            
        self.client = MemoryClient(api_key=api_key)
        self.loop = asyncio.get_event_loop()
    
    async def add_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extract information from memory to format for Mem0
            memory_type = memory.get("type", "status_update")
            summary = memory.get("summary", "")
            raw_content = memory.get("raw_content", summary)
            author = memory.get("participants", ["unknown"])[0].replace("@", "")
            
            # Use channel_id as user_id - THIS IS IMPORTANT FOR CONSISTENCY
            user_id = memory.get("channel_id", "default_channel")
            
            # Store the user_id in memory for debugging purposes
            logger.info(f"Storing memory with user_id: {user_id}")
            
            # Format as messages for Mem0
            messages = [
                {"role": "user", "content": raw_content},
                {"role": "assistant", "content": f"I've recorded this {memory_type}: {summary}"}
            ]
            
            # Run the synchronous Mem0 client in a thread pool
            return await self.loop.run_in_executor(
                None, 
                lambda: self.client.add(
                    messages, 
                    user_id=user_id,
                    app_id="team_memory_bot"
                )
            )
        except Exception as e:
            logger.error(f"Error storing memory in Mem0: {e}", exc_info=True)
            raise
    
    async def search_memories(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query memories from the Mem0 system.
        
        Args:
            query (str): Natural language query
            user_id (str, optional): User ID to limit search scope
            
        Returns:
            Dict[str, Any]: Response with matching memories
        """
        try:
            # Always use a list of user_ids to search
            search_user_ids = []
            
            # If specific user_id provided, search that
            if user_id:
                search_user_ids.append(user_id)
                logger.info(f"Searching memory for user_id: {user_id}")
            else:
                # If no user_id provided, search all monitored channels
                from config import MONITORED_CHANNELS
                search_user_ids.extend(MONITORED_CHANNELS)
                logger.info(f"Searching memory across all channels: {search_user_ids}")
            
            # Store all found memories
            all_memories = []
            
            # Search each user_id separately
            for current_user_id in search_user_ids:
                try:
                    logger.info(f"Mem0 search with exact parameters: query='{query}', user_id='{current_user_id}', app_id='team_memory_bot'")
                    logger.info(f"Querying Mem0 with: query='{query}', user_id='{current_user_id}'")
                    
                    # Call the Mem0 API
                    results = await self.loop.run_in_executor(
                        None,
                        lambda: self.client.search(
                            query, 
                            user_id=current_user_id,
                            app_id="team_memory_bot"
                        )
                    )
                    
                    # Log what we found
                    if results:
                        logger.info(f"Found {len(results)} memories for user_id {current_user_id}")
                        # Add each memory's summary for debugging
                        for i, memory in enumerate(results):
                            if isinstance(memory, dict):
                                if 'memory' in memory:
                                    logger.info(f"Memory {i} FULL content: {memory['memory']}")
                        
                        all_memories.extend(results)
                    else:
                        logger.info(f"No memories found in user_id {current_user_id}")
                        
                except Exception as e:
                    logger.warning(f"Error searching user_id {current_user_id}: {e}")
                    continue
            
            # Format the results to match our internal structure
            formatted_results = {
                "memories": []
            }
            
            # Log total memories found
            logger.info(f"Total memories found across all channels: {len(all_memories)}")
            
            # Process each memory with careful error handling
            for i, memory_data in enumerate(all_memories):
                try:
                    # Skip None values
                    if memory_data is None:
                        continue
                        
                    # Get content safely - MORE DETAILED LOGGING
                    content = ""
                    if isinstance(memory_data, dict):
                        logger.info(f"Memory data keys: {memory_data.keys()}")
                        
                        # Try different possible content fields, starting with 'memory'
                        if 'memory' in memory_data:
                            content = memory_data['memory']
                            logger.info(f"Found memory field with value: '{content}'")
                        elif 'content' in memory_data:
                            content = memory_data['content']
                            logger.info(f"Found content field with value: '{content}'")
                        elif 'text' in memory_data:
                            content = memory_data['text']
                            logger.info(f"Found text field with value: '{content}'")
                        else:
                            # Log the entire memory data to see what fields are available
                            logger.info(f"Memory data structure: {memory_data}")
                            
                    # Get metadata safely
                    metadata = {}
                    if isinstance(memory_data, dict) and isinstance(memory_data.get("metadata"), dict):
                        metadata = memory_data.get("metadata", {})
                    
                    # Create memory object with safe defaults
                    memory = {
                        "id": str(uuid.uuid4()),  # Default ID
                        "type": "memory",         # Default type
                        "summary": content,       # Now using the content from memory field
                        "timestamp": "",
                        "context": "",
                        "participants": ["unknown"],
                    }
                    
                    # Try to get ID from result
                    if isinstance(memory_data, dict) and memory_data.get("id"):
                        memory["id"] = memory_data.get("id")
                    
                    # Update fields from metadata if available
                    if metadata:
                        memory["type"] = metadata.get("type", "memory")
                        memory["timestamp"] = metadata.get("timestamp", "")
                        memory["context"] = metadata.get("context", "")
                        
                        # Handle participants
                        if metadata.get("author"):
                            memory["participants"] = [metadata.get("author")]
                    
                    # Add to results
                    formatted_results["memories"].append(memory)
                    
                except Exception as e:
                    logger.warning(f"Error processing memory {i}: {e}")
                    continue
            
            logger.info(f"Successfully formatted {len(formatted_results['memories'])} memories")
            
            # Log the first few formatted memories for debugging
            for i, memory in enumerate(formatted_results["memories"][:3]):
                logger.info(f"Formatted memory {i} FULL content: type={memory['type']}, summary={memory['summary']}")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying memories from Mem0: {e}", exc_info=True)
            return {"memories": []}

# Global client instance
_mem0_client = None

def get_mem0_client() -> Mem0Wrapper:
    """
    Get or create the global Mem0 client instance.
    
    Returns:
        Mem0Wrapper: The Mem0 client wrapper instance
    """
    global _mem0_client
    if _mem0_client is None:
        _mem0_client = Mem0Wrapper(api_key=MEM0_API_KEY)
    return _mem0_client

# Module-level functions that can be imported directly

async def store_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store a memory in the Mem0 system.
    
    Args:
        memory (Dict[str, Any]): Memory data to store
        
    Returns:
        Dict[str, Any]: Response from the Mem0 API
    """
    client = get_mem0_client()
    return await client.add_memory(memory)

async def query_memories(query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Query memories from the Mem0 system.
    
    Args:
        query (str): Natural language query
        user_id (str, optional): Optional user ID to limit search to a specific channel
        
    Returns:
        Dict[str, Any]: Response with matching memories
    """
    client = get_mem0_client()
    return await client.search_memories(query, user_id=user_id)