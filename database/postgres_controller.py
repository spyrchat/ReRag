import os
import uuid
import datetime
import logging
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import OperationalError
from logs.utils.logger import get_logger
logger = get_logger("postgres_controller")

# Load environment variables from .env
load_dotenv(override=True)

# Define SQLAlchemy Base
Base = declarative_base()


class ImageAsset(Base):
    """
    ORM model for storing image assets metadata in the 'image_assets' table.
    """
    __tablename__ = 'image_assets'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    caption = Column(Text)
    extracted_text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class TableAsset(Base):
    """
    ORM model for storing table assets metadata in the 'table_assets' table.
    """
    __tablename__ = 'table_assets'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    table_json = Column(Text, nullable=False)
    caption = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PostgresController:
    """
    Controller for managing PostgreSQL database connections and asset insertions.
    Loads DB config from a config dict (or environment variables), handles session management,
    and provides methods for inserting image/table asset metadata.
    """
    engine = None
    SessionLocal = None

    def __init__(self, db_config: dict = None):
        """
        Initialize the database connection using config dict or environment variables.
        Raises ValueError if configuration is missing, or ConnectionError if DB is unreachable.
        """
        db_config = db_config or {}

        # Prefer config dict; fallback to env vars if not set
        user = db_config.get('user') or os.getenv('POSTGRES_USER')
        password = db_config.get('password') or os.getenv('POSTGRES_PASSWORD')
        host = db_config.get('host') or os.getenv('POSTGRES_HOST')
        port = db_config.get('port') or os.getenv('POSTGRES_PORT')
        database = db_config.get('database') or os.getenv('POSTGRES_DB')

        if not all([user, password, host, port, database]):
            logger.error(
                "One or more required Postgres configuration variables are missing.")
            raise ValueError(
                "One or more required Postgres configuration variables are missing.")

        connection_str = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
        logger.info(f"Connecting to Postgres: {connection_str}")

        try:
            self.engine = create_engine(connection_str)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine)
            logger.info(
                "PostgreSQL engine and tables initialized successfully.")
        except OperationalError as e:
            logger.error(f"Failed to connect to the database: {e}")

    def get_session(self) -> Session:
        """
        Create and return a new SQLAlchemy session.
        Caller is responsible for closing or using with-statement.
        """
        return self.SessionLocal()

    def insert_image_asset(self, doc_id: str, page_number: int, file_path: str,
                           caption: str = None, extracted_text: str = None):
        """
        Insert a new image asset record into the database.
        Args:
            doc_id (str): Document identifier.
            page_number (int): Page number where image appears.
            file_path (str): File path of the stored image.
            caption (str, optional): Caption text.
            extracted_text (str, optional): Extracted OCR text from image.
        """
        with self.get_session() as session:
            asset = ImageAsset(
                doc_id=doc_id,
                page_number=page_number,
                file_path=file_path,
                caption=caption,
                extracted_text=extracted_text,
            )
            session.add(asset)
            session.commit()
            logger.info(
                f"Inserted ImageAsset for doc_id={doc_id}, page_number={page_number}")

    def insert_table_asset(self, doc_id: str, page_number: int, table_json: str,
                           caption: str = None):
        """
        Insert a new table asset record into the database.
        Args:
            doc_id (str): Document identifier.
            page_number (int): Page number where table appears.
            table_json (str): Serialized JSON for table contents.
            caption (str, optional): Caption text.
        """
        with self.get_session() as session:
            asset = TableAsset(
                doc_id=doc_id,
                page_number=page_number,
                table_json=table_json,
                caption=caption,
            )
            session.add(asset)
            session.commit()
            logger.info(
                f"Inserted TableAsset for doc_id={doc_id}, page_number={page_number}")


if __name__ == "__main__":
    """
    Main execution block for connectivity testing.
    Attempts to connect to PostgreSQL and run a simple test query.
    """
    try:
        controller = PostgresController()
        with controller.get_session() as session:
            session.execute(text("SELECT 1"))
            logger.info("PostgreSQL connection successful. Tables are ready.")
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
