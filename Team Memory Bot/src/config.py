import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Discord configuration
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
MONITORED_CHANNELS = os.getenv("MONITORED_CHANNELS", "").split(",")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:latest")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Vector database configuration
VECTOR_DB = os.getenv("VECTOR_DB", "chromadb")
CHROMADB_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "./chroma_db")

# Database configuration - using SQLite by default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///team_memory.db")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").split("#")[0].strip()

# Mem0 Memory Engine
MEM0_API_URL = os.getenv("MEM0_API_URL", "")
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

# Memory types
MEMORY_TYPES = ["decision", "blocker", "status_update", "milestone", "question", "answer"]

# Command prefix
COMMAND_PREFIX = "/"

# RAG configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 4096
TOP_K_RESULTS = 5