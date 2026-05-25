from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:sharika123@localhost:5432/scada_db"

engine = create_engine(DB_URL)