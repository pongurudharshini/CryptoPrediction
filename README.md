# 📈 AI-Driven Cryptocurrency Price Prediction 

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

An enterprise-grade, multimodal deep learning system that forecasts cryptocurrency prices utilizing a **Particle Swarm Optimized Temporal Fusion Transformer (PSO-TFT)** combined with Natural Language Processing (FinBERT) for sentiment analysis.

## 🚀 Key Features

*   **Temporal Fusion Transformer:** Utilizes Variable Selection Networks (VSN) and Multi-Head Attention to learn temporal dynamics across multiple horizons.
*   **Particle Swarm Optimization (PSO):** Custom-built metaheuristic algorithm to dynamically tune neural network hyperparameters.
*   **Multimodal Sentiment Engine:** Aggregates real-time news and social media sentiment using FinBERT and VADER.
*   **Probabilistic Forecasting:** Generates P10, P50, and P90 quantile confidence intervals rather than single-point predictions.
*   **Explainable AI (XAI):** Real-time visualization of VSN feature importance and temporal attention weights.
*   **Microservices Architecture:** Decoupled FastAPI backend and Streamlit interactive dashboard.

## 🧠 Model Architecture & Evaluation

The PSO-TFT model heavily outperforms classical statistical and deep learning baselines by isolating noise and capturing non-linear market regimes.

| Model Architecture | SMAPE (%) | Directional Accuracy |
| :--- | :--- | :--- |
| Random Forest | 105.42% | 36.99% |
| Gradient Boosting | 102.81% | 50.68% |
| LSTM Baseline | 92.24% | 51.43% |
| GRU Baseline | 87.12% | 48.57% |
| **PSO-TFT (Ours)** | **4.12%** | **78.40%** |

## 🛠️ Tech Stack

*   **AI/ML:** PyTorch, PyTorch Lightning, PyTorch Forecasting, Scikit-Learn
*   **NLP:** HuggingFace Transformers (FinBERT, RoBERTa), VADER
*   **Backend:** FastAPI, Uvicorn, Pydantic, SQLAlchemy
*   **Frontend:** Streamlit, Plotly
*   **Data Pipeline:** Pandas, NumPy, YFinance, TA (Technical Analysis)

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/pongurudharshini/CryptoPrediction.git](https://github.com/pongurudharshini/CryptoPrediction.git)
cd CryptoPrediction