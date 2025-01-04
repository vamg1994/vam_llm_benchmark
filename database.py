"""
Database configuration and connection management.
This module handles PostgreSQL database initialization and connection pooling
using SQLAlchemy. It provides a clean interface for database operations
throughout the application.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, DBAPIError


# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get database URL from environment with validation
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL is None:
    logger.error("DATABASE_URL environment variable is not set")
    raise ValueError("DATABASE_URL environment variable is not set")

# Convert legacy postgres:// URLs to postgresql://
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://')

# Create base class for declarative models
Base = declarative_base()

# Configure database engine with optimal settings for Supabase PostgreSQL
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,     # Enable connection health checks
        pool_size=5,            # Reduced pool size to prevent connection issues
        max_overflow=10,        # Reduced max overflow connections
        pool_timeout=30,        # Connection timeout in seconds
        pool_recycle=1800,      # Recycle connections after 30 minutes
        echo=False             # Disable SQL query logging for production
    )

    # Create session factory
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    logger.info("Successfully initialized Supabase PostgreSQL connection")
except Exception as e:
    logger.error(f"Failed to initialize database connection: {e}")
    raise

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Provides automatic cleanup of database sessions and proper error handling.

    Yields:
        Session: SQLAlchemy session object

    Raises:
        SQLAlchemyError: If database operations fail
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database operation failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Add event listener for connection monitoring
@event.listens_for(Engine, "connect")
def connect(dbapi_connection, connection_record):
    """Log successful database connections"""
    logger.info("New database connection established to Supabase PostgreSQL")

def verify_database_connection():
    """
    Verify database connection is working properly.

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            logger.info("Database connection verified successfully")
            return True
    except Exception as e:
        logger.error(f"Database connection verification failed: {e}")
        return False