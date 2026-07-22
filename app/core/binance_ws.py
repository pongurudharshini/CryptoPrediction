"""
Binance WebSocket Manager.
Maintains a continuous, low-latency connection to Binance streams for real-time AI predictions.
"""

import asyncio
import json
import logging
import websockets
from typing import Callable, List
from config.settings import settings

logger = logging.getLogger(__name__)

class BinanceWebSocketManager:
    """Manages asynchronous WebSocket connections to Binance."""
    
    def __init__(self, symbols: List[str], callback: Callable):
        """
        Args:
            symbols: List of coin pairs (e.g., ['btcusdt', 'ethusdt']).
            callback: Async function to process incoming JSON data.
        """
        # Binance requires lowercase symbols for WS streams
        self.symbols = [s.lower() for s in symbols]
        self.callback = callback
        self.is_running = False
        
        # Create a combined stream URL
        # E.g., wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/ethusdt@ticker
        streams = "/".join([f"{sym}@ticker" for sym in self.symbols])
        self.ws_url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    async def start(self):
        """Connects to WebSocket and listens continuously."""
        self.is_running = True
        logger.info(f"Connecting to Binance WS: {self.ws_url}")
        
        while self.is_running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("WebSocket connected successfully!")
                    while self.is_running:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        # Extract the actual ticker payload from the combined stream format
                        if 'data' in data:
                            await self.callback(data['data'])
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket closed unexpectedly. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket Error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """Gracefully stops the WebSocket loop."""
        self.is_running = False
        logger.info("WebSocket shutdown requested.")