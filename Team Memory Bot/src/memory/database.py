import logging
import os
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from config import DATABASE_URL

# Set up logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define database models
class Memory(Base):
    """Memory model for storing memory records."""
    __tablename__ = "memories"
    
    id = Column(String(36), primary_key=True)
    type = Column(String(50), index=True)
    summary = Column(Text)
    context = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    participants = Column(Text, nullable=True)  # Comma-separated participants
    
    # Relationship to metadata
    metadata = relationship("MemoryMetadata", back_populates="memory", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert memory to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "summary": self.summary,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "participants": self.participants.split(",") if self.participants else []
        }

class MemoryMetadata(Base):
    """Metadata for memory records."""
    __tablename__ = "memory_metadata"
    
    id = Column(Integer, primary_key=True)
    memory_id = Column(String(36), ForeignKey("memories.id"))
    key = Column(String(100))
    value = Column(Text)
    
    # Relationship to memory
    memory = relationship("Memory", back_populates="metadata")

def init_db():
    """Initialize the database, creating tables if they don't exist."""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

# Initialize database on module load
if __name__ == "__main__":
    init_db()
    logger.info("Database initialized during module load")