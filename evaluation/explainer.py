"""
TFT Explainability and Visualization Engine.
Extracts Feature Importance, Attention Heatmaps, and Prediction Intervals.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = ROOT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


class ModelExplainer:
    """Generates explainability plots for the Temporal Fusion Transformer."""

    def __init__(self, model: TemporalFusionTransformer, dataset: TimeSeriesDataSet):
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _get_evaluation_batch(self):
        """Helper to get a batch of validation data moved to the target device."""
        val_dataloader = self.dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
        x, _ = next(iter(val_dataloader))
        # Move tensor values in the dictionary to the target device
        x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        return x

    def plot_prediction_intervals(self, dataloader: DataLoader, num_plots: int = 1):
        """Plots the actual vs predicted prices with confidence intervals."""
        logger.info("Generating Prediction Interval Plots...")
        
        x, _ = next(iter(dataloader))
        x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        
        with torch.no_grad():
            predictions = self.model(x)
        
        for idx in range(min(num_plots, x['encoder_target'].shape[0])):
            fig, ax = plt.subplots(figsize=(10, 5))
            self.model.plot_prediction(x, predictions, idx=idx, ax=ax, add_loss_to_title=True)
            ax.set_title(f"TFT Forecast with Confidence Intervals (Sample {idx+1})", fontsize=14)
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Normalized Price")
            plt.tight_layout()
            
            save_path = VIS_DIR / f"prediction_intervals_{idx}.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved: {save_path}")

    def plot_feature_importance(self):
        """
        Manually extracts and plots Variable Selection Network (VSN) weights.
        Custom implementation to prevent library indexing errors with single static variables.
        """
        logger.info("Extracting Feature Importance (VSN Weights)...")
        
        x = self._get_evaluation_batch()
        with torch.no_grad():
            output = self.model(x)
            interpretation = self.model.interpret_output(output)

        # 1. Encoder Variable Importance
        if "encoder_variables" in interpretation:
            # Safely average dimensions and force to 1D array to prevent 0-dimensional IndexErrors
            enc_weights = interpretation["encoder_variables"].mean(dim=[0, 1]).cpu().numpy()
            enc_weights = np.atleast_1d(enc_weights)
            
            enc_names = self.model.encoder_variables
            
            # If names don't match weights length (rare edge case), truncate to match
            min_len = min(len(enc_weights), len(enc_names))
            enc_weights = enc_weights[:min_len]
            enc_names = enc_names[:min_len]
            
            # Sort by importance ascending for horizontal bar chart
            sorted_idx = np.argsort(enc_weights)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(sorted_idx)), enc_weights[sorted_idx] * 100, color='skyblue', edgecolor='black')
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([enc_names[i] for i in sorted_idx], fontsize=11)
            ax.set_xlabel("Variable Importance (%)", fontsize=12)
            ax.set_title("Encoder Feature Importance (Variable Selection Network)", fontsize=14, pad=15)
            ax.grid(True, linestyle='--', alpha=0.5, axis='x')
            
            # Add percentage labels to bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                        va='center', ha='left', fontsize=10)

            plt.tight_layout()
            save_path = VIS_DIR / "feature_importance_encoder.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved: {save_path}")

        # 2. Decoder Variable Importance (if applicable)
        if "decoder_variables" in interpretation and interpretation["decoder_variables"].numel() > 0:
            dec_weights = interpretation["decoder_variables"].mean(dim=[0, 1]).cpu().numpy()
            dec_weights = np.atleast_1d(dec_weights)
            
            dec_names = self.model.decoder_variables
            min_len = min(len(dec_weights), len(dec_names))
            dec_weights = dec_weights[:min_len]
            dec_names = dec_names[:min_len]
            
            sorted_idx = np.argsort(dec_weights)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(range(len(sorted_idx)), dec_weights[sorted_idx] * 100, color='lightgreen', edgecolor='black')
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([dec_names[i] for i in sorted_idx], fontsize=11)
            ax.set_xlabel("Variable Importance (%)", fontsize=12)
            ax.set_title("Decoder Feature Importance", fontsize=14, pad=15)
            ax.grid(True, linestyle='--', alpha=0.5, axis='x')
            
            plt.tight_layout()
            save_path = VIS_DIR / "feature_importance_decoder.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved: {save_path}")

    def plot_attention_heatmap(self):
        """Plots Multi-Head Attention weights across past time steps."""
        logger.info("Extracting Temporal Attention Weights...")
        
        x = self._get_evaluation_batch()
        with torch.no_grad():
            output = self.model(x)
            interpretation = self.model.interpret_output(output)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Average attention weights across batches
        attention_mean = interpretation["attention"].mean(0).cpu().numpy()
        attention_mean = np.atleast_1d(attention_mean)
        
        ax.plot(range(-len(attention_mean), 0), attention_mean, marker='o', color='darkblue', linewidth=2)
        ax.set_title("Temporal Attention Weights over Time Horizon", fontsize=14, pad=15)
        ax.set_xlabel("Days in Past relative to Prediction Point", fontsize=12)
        ax.set_ylabel("Attention Weight", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = VIS_DIR / "attention_weights.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved: {save_path}")