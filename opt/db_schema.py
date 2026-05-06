import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Explicitly load .env file
load_dotenv()

# Get DB_URL from environment or construct it from parts
# We force 127.0.0.1 to avoid IPv1/IPv6 localhost confusion
DB_USER = os.getenv("DB_USER", "quant_admin")
DB_PASS = os.getenv("DB_PASSWORD", "quant_password")
DB_NAME = os.getenv("DB_NAME", "btc_quant_db")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'btc_usd_ohlcv'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('idx_timestamp_timeframe', 'timestamp', 'timeframe', unique=True),
    )

class TickData(Base):
    __tablename__ = 'btc_usd_ticks'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    side = Column(String(4), nullable=False)

class TradeLog(Base):
    __tablename__ = 'trade_executions'
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String(10))
    price = Column(Float)
    size = Column(Float)
    confidence_score = Column(Float)
    regime = Column(String(20))

def init_db():
    try:
        Base.metadata.create_all(engine)
        print("Database tables initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")

def get_session():
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    init_db()
