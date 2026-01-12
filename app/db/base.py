"""
Database Manager
SQLAlchemy database initialization and session management
"""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging

from .models import Base, VoiceModel, AudioConfig, ErrorLog, RuntimeMetric

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages database connection and sessions
    """
    
    def __init__(self, db_path: Path):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.engine = None
        self.SessionLocal = None
        self._session = None
    
    def initialize(self):
        """Initialize database and create tables"""
        try:
            # Create database directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create engine with SQLite
            database_url = f"sqlite:///{self.db_path.absolute()}"
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """
        Get database session
        
        Returns:
            SQLAlchemy session
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.SessionLocal()
    
    def dispose(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


