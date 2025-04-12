import chromadb

# Initialize client with persistence
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection
collection = client.create_collection(
    name="team_memory",
    metadata={"hnsw:space": "cosine"}
)

print("Collection 'team_memory' created successfully!")