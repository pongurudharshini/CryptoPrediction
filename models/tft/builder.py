"""
Temporal Fusion Transformer (TFT) Builder.
Constructs the TFT architecture with Quantile Loss for confidence intervals.
"""

from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import TimeSeriesDataSet
import logging

logger = logging.getLogger(__name__)

class TFTBuilder:
    """Builds and configures the Temporal Fusion Transformer."""

    @staticmethod
    def build_model(
        training_dataset: TimeSeriesDataSet,
        learning_rate: float = 0.03,
        hidden_size: int = 16,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        hidden_continuous_size: int = 8
    ) -> TemporalFusionTransformer:
        """
        Initializes the TFT model based on the dataset structure.
        
        Args:
            training_dataset: The PyTorch Forecasting dataset.
            learning_rate: Optimizer learning rate.
            hidden_size: Number of neurons in hidden layers.
            attention_head_size: Number of attention heads.
            dropout: Dropout rate to prevent overfitting.
            hidden_continuous_size: Hidden size for processing continuous variables.
            
        Returns:
            Configured TemporalFusionTransformer model.
        """
        logger.info("Initializing Temporal Fusion Transformer Architecture...")
        
        # QuantileLoss allows us to predict a range (e.g., 10%, 50%, 90% confidence)
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=loss_fn,
            log_interval=10,  # Logs metrics every 10 batches
            reduce_on_plateau_patience=4, # Reduces LR if validation loss plateaus
        )
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"TFT Model built successfully with {total_params:,} trainable parameters.")
        
        return model