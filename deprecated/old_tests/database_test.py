import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )
    print("psycopg2 connection successful!")
    conn.close()
except Exception as e:
    print(f"psycopg2 connection failed: {e}")
