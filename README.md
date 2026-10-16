# 📈 AI-Driven Cryptocurrency Price Prediction | PSO-TFT Predictor

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An institutional-grade, full-stack cryptocurrency analytics dashboard powered by **Particle Swarm Optimized Temporal Fusion Transformers (PSO-TFT)** and **Spatio-Temporal Graph Neural Networks (ST-GNN)**. 

This platform tracks 100 top-tier digital assets in real-time, providing deep quantitative insights, macroeconomic correlations, and actionable AI-driven trading signals wrapped in a premium glassmorphism interface.

---

## ✨ Core Features

### 🧠 Advanced AI & Machine Learning
*   **PSO-TFT Forecasting:** Multi-horizon price predictions with 90% confidence interval quantile bounds (P10 Stop-Loss / P90 Take-Profit).
*   **ST-GNN Spillover Network:** Analyzes cross-currency lead-lag contagion and dynamic systemic market beta.
*   **Multimodal Sentiment Engine:** Cross-attention NLP using FinBERT to scrape live news and social media to calculate real-time Fear & Greed weightings.
*   **Continuous MLOps:** Live data drift telemetry (PSI scoring) and one-click automated model retraining pipelines.

### 📊 Quantitative Institutional Engines
*   **Portfolio Optimization:** Markowitz Mean-Variance Efficient Frontier combined with Black-Litterman Bayesian View Synthesis.
*   **Monte Carlo Stress Testing:** 1,000 stochastic Geometric Brownian Motion (GBM) simulated price paths to calculate VaR (Value at Risk) and Expected Shortfall (CVaR).
*   **Derivatives Liquidation Heatmap:** Tracks perpetual futures funding rates, long/short leverage imbalances, and visualizes multi-million dollar liquidation clusters.
*   **Macro-Economic Monitor:** Correlates crypto regimes against real-time U.S. Dollar Index (DXY), 10Y Treasury Yields, S&P 500 beta, and Federal Reserve Net Liquidity.

### 🛠️ Platform & UI
*   **Premium Glassmorphism UI:** A heavily customized, visually striking dark-mode interface.
*   **Real-Time Sub-Second Feeds:** Integrates live Binance WebSockets alongside `yfinance` API fallbacks.
*   **Automated Webhook Bots:** Asynchronous trading alerts dispatched instantly to Discord channels and Telegram mobile apps.
*   **TFT RAG Copilot:** A locally run, omniscient Retrieval-Augmented Generation AI assistant with Speech-to-Text capabilities for hands-free querying.

---

## 🏗️ Technology Stack

*   **Frontend UI:** [Streamlit](https://streamlit.io/) (with custom CSS/HTML injection)
*   **Backend Engine:** [FastAPI](https://fastapi.tiangolo.com/) (RESTful API architecture)
*   **Database:** SQLite / SQLAlchemy (Local Event Bus & User Auth)
*   **Data Pipelines:** Pandas, NumPy, yfinance, Binance WebSockets
*   **Machine Learning:** PyTorch, Scikit-learn, SciPy, HuggingFace (FinBERT)
*   **Interactive Visualization:** Plotly Graph Objects & Plotly Express
*   **Export & Reporting:** ReportLab (Automated PDF Generation)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/PSO-TFT-Crypto-Dashboard.git](https://github.com/YOUR_USERNAME/PSO-TFT-Crypto-Dashboard.git)
cd PSO-TFT-Crypto-Dashboard


Step 2: Configure Environment Variables
Create a file named .env in the root directory (next to README.md) and add your secure keys:

Code snippet
SECRET_KEY=your_secure_bcrypt_key_here
PORT=8501
Option A: Run Locally (Standard Python Method)
1. Create and activate a virtual environment:

Bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
2. Install all required dependencies:

Bash
pip install -r requirements.txt
3. Start the Application (Requires 2 Terminal Windows):

Because this is a decoupled architecture, you must run the backend and frontend simultaneously.

Terminal 1 (Start the FastAPI Backend):
Ensure your virtual environment is active, then run:

Bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
(Your backend is now running at http://127.0.0.1:8000/docs)

Terminal 2 (Start the Streamlit Frontend):
Open a second terminal, activate the virtual environment, and run:

Bash
streamlit run frontend/dashboard.py
The AI Crypto Dashboard will automatically open in your browser at http://localhost:8501.

Option B: Run Locally (Docker Method - Recommended)
If you have Docker installed, you can launch the entire ecosystem (database, backend, and frontend) with a single command.

Make sure your Docker Desktop application is running, then execute:

Bash
docker-compose up --build
(To run it quietly in the background, use docker-compose up --build -d)

Wait for the containers to initialize. Once built, open your web browser and go to:
http://localhost:8501

🔔 Setting Up Automated Bots (Discord & Telegram)
The platform supports live automated alerts for price targets, RSI washouts, and portfolio drift.

Discord: Create a Webhook in your server settings (Integrations > Webhooks) and paste the generated URL into the dashboard's "Alerts & Automated Bots" tab.

Telegram: Message @BotFather on Telegram to create a bot and get an HTTP Token. Then, message @userinfobot to get your Chat ID. You must send /start to your new bot before it can message you. Enter these credentials into the dashboard to test the connection.

⚠️ Disclaimer
This software is for educational and research purposes only. Cryptocurrency trading involves severe market volatility and significant risk of capital loss. The PSO-TFT price predictions, stop-loss floors, and portfolio optimization suggestions generated by this repository do not constitute financial advice. Never deploy capital you cannot afford to lose.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.