from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import OperationalError
import uuid
import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

# Define SQLAlchemy Base
Base = declarative_base()

# Define ImageAsset table


class ImageAsset(Base):
    __tablename__ = 'image_assets'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    caption = Column(Text)
    extracted_text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Define TableAsset table


class TableAsset(Base):
    __tablename__ = 'table_assets'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    table_json = Column(Text, nullable=False)
    caption = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Postgres controller class


class PostgresController:
    engine = None
    SessionLocal = None

    def __init__(self):
        # Read DB settings from environment variables
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        host = os.getenv('POSTGRES_HOST')
        port = os.getenv('POSTGRES_PORT')
        database = os.getenv('POSTGRES_DB')

        # Check for missing variables
        if not all([user, password, host, port, database]):
            raise ValueError(
                "One or more required Postgres environment variables are missing.")

        # Build connection string (no spaces!)
        connection_str = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
        print(f"Connecting to: {connection_str}")

        try:
            self.engine = create_engine(connection_str)
            Base.metadata.create_all(self.engine)
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine)
        except OperationalError as e:
            raise ConnectionError(f"Failed to connect to the database: {e}")

    def get_session(self) -> Session:
        return self.SessionLocal()


if __name__ == "__main__":
    try:
        controller = PostgresController()
        with controller.get_session() as session:
            session.execute(text("SELECT 1"))
            print("PostgreSQL connection successful. Tables are ready.")
    except Exception as e:
        print(f"Connection failed: {str(e)}")
