"""
Database ORM Models.
Defines the schema for storing historical cryptocurrency market data.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from database.session import Base
from datetime import datetime

class OHLCVData(Base):
    __tablename__ = "ohlcv_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    market_cap = Column(Float, nullable=True) # Optional, mostly from CoinGecko/YFinance

    # Create a composite index to quickly query specific coins within date ranges
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp', unique=True),
    )

    def __repr__(self):
        return f"<OHLCVData(symbol={self.symbol}, date={self.timestamp}, close={self.close})>"