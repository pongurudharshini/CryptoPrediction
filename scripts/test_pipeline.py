"""
Data Pipeline Integration Test.
Verifies DB creation, REST API fetching, and WebSocket streaming.
"""

import asyncio
from database.session import engine, Base
from database.models import OHLCVData
from data_pipeline.ingest.historical import HistoricalDataFetcher
from app.core.binance_ws import BinanceWebSocketManager

async def live_price_callback(data: dict):
    """Callback function for WebSocket stream."""
    symbol = data.get('s')
    price = data.get('c') # 'c' is the live close price in Binance ticker stream
    print(f"[LIVE WebSocket] {symbol} Price: ${price}")

async def main():
    print("=== 1. Initializing Database ===")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.\n")

    print("=== 2. Fetching Historical Data (YFinance) ===")
    df_btc = HistoricalDataFetcher.fetch_yfinance("BTC-USD", period="5d")
    print("YFinance Data:\n", df_btc.tail(3), "\n")

    print("=== 3. Fetching Historical Data (Binance REST) ===")
    df_eth = HistoricalDataFetcher.fetch_binance("ETHUSDT", limit=3)
    print("Binance Data:\n", df_eth, "\n")

    print("=== 4. Testing Binance WebSocket (Streaming for 10 seconds) ===")
    ws_manager = BinanceWebSocketManager(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], 
        callback=live_price_callback
    )
    
    # Run WS in background
    ws_task = asyncio.create_task(ws_manager.start())
    
    # Let it run for 10 seconds to observe output
    await asyncio.sleep(10)
    
    # Shut it down
    ws_manager.stop()
    ws_task.cancel()
    print("\nData Pipeline test completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())