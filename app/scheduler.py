"""
Background Scheduler for AI Crypto Predictor.
Monitors live prices and triggers In-App Notifications (Bell Icon) 
when user-defined target buy prices are reached.
"""

import sys
import os
import time
from datetime import datetime
import yfinance as yf

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db import SessionLocal, AlertPreference, InAppNotification

def check_alerts_and_notify():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running market check for active alerts...")
    db = SessionLocal()
    preferences = db.query(AlertPreference).all()

    if not preferences:
        print("No active alerts to monitor in the database.")
        db.close()
        return

    for pref in preferences:
        try:
            # Fetch the latest live price using yfinance
            ticker = yf.Ticker(pref.symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                continue
                
            current_price = hist['Close'].iloc[-1]

            # Check if the price has dropped to or below the user's target buy price
            if pref.target_buy_price and current_price <= pref.target_buy_price:
                print(f"🚨 ALERT TRIGGERED: {pref.symbol} hit ${current_price:,.2f} for {pref.user_email}")
                
                # 1. Create the In-App Notification (Bell Icon)
                notification = InAppNotification(
                    user_email=pref.user_email,
                    title=f"🚨 Price Alert: {pref.symbol} Target Reached!",
                    message=f"The price of {pref.symbol} has dropped to **${current_price:,.2f}**, hitting your target buy entry of **${pref.target_buy_price:,.2f}**."
                )
                db.add(notification)
                
                # 2. Delete the alert preference so it doesn't spam the user every 60 seconds
                db.delete(pref)
                db.commit()

        except Exception as e:
            print(f"Error checking {pref.symbol}: {e}")

    db.close()
    print("Market check complete. Waiting for next cycle...")

if __name__ == "__main__":
    print("🚀 Starting Internal Notification Background Scheduler...")
    print("Press Ctrl+C to stop.")
    
    # Run in a continuous loop every 60 seconds
    while True:
        try:
            check_alerts_and_notify()
            time.sleep(60) # Wait 60 seconds before checking prices again
        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped.")
            break
        except Exception as e:
            print(f"Scheduler encountered an error: {e}")
            time.sleep(60)