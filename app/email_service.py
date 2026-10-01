"""
Email Notification Service using Resend API.
Handles automated welcome messages and safe-buy AI reports.
"""

import os
import resend
from dotenv import load_dotenv

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")

def send_welcome_email(to_email: str, user_name: str):
    """Sends a professional HTML onboarding email via Resend API."""
    if not resend.api_key:
        print("⚠️ RESEND_API_KEY not found. Skipping email.")
        return False

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #0e1117; color: #f0f6fc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #161b22; border-radius: 10px; border: 1px solid #30363d; padding: 25px;">
            <h2 style="color: #00F0FF; margin-top: 0;">🚀 Welcome to PSO-TFT Crypto Predictor!</h2>
            <p>Hi <b>{user_name}</b>,</p>
            <p>Your account has been successfully created. You now have access to institutional-grade AI market forecasting.</p>
            
            <div style="background-color: #0e1117; border-left: 4px solid #3fb950; padding: 12px; margin: 20px 0; border-radius: 4px;">
                <h4 style="margin: 0 0 8px 0; color: #3fb950;">💡 AI Safe-Buy Guidelines</h4>
                <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #8b949e;">
                    <li>Risk rule: Never allocate more than 2%–5% per trade.</li>
                    <li>Always check P10 Lower Bounds for protective stop-loss levels.</li>
                    <li>Verify market sentiment indicators before trade execution.</li>
                </ul>
            </div>
            
            <p style="font-size: 13px; color: #8b949e;">Happy Trading,<br><b>The PSO-TFT Quantitative Team</b></p>
        </div>
    </body>
    </html>
    """

    try:
        params: resend.Emails.SendParams = {
            "from": "PSO-TFT Predictor <onboarding@resend.dev>",
            "to": [to_email],
            "subject": "Welcome to PSO-TFT AI Predictor! 📈",
            "html": html_content,
        }
        resend.Emails.send(params)
        print(f"✅ Welcome email successfully sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Error sending email via Resend: {e}")
        return False