"""
PyTorch Lightning Training Loop.
Handles GPU/MPS/CPU hardware detection, Early Stopping, and Model Checkpointing.
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent

class TFTTrainer:
    """Configures the PyTorch Lightning Trainer."""

    @staticmethod
    def get_trainer(max_epochs: int = 30, use_gpu: bool = True) -> pl.Trainer:
        """
        Sets up the trainer with callbacks for Early Stopping, Checkpointing, and Logging.
        Safely falls back to CPU if no GPU hardware is detected.
        """
        # 1. Early Stopping: Stop training if validation loss doesn't improve for 5 epochs
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=5,
            verbose=True,
            mode="min"
        )

        # 2. Checkpointing: Save the best model automatically
        checkpoint_dir = ROOT_DIR / "saved_models" / "tft_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        # 3. Logger setup with automatic fallback
        log_dir = ROOT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from lightning.pytorch.loggers import TensorBoardLogger
            tb_logger = TensorBoardLogger(save_dir=log_dir, name="tft_training")
            logger.info("Using TensorBoardLogger for metrics tracking.")
        except Exception as e:
            logger.warning(f"TensorBoard unavailable ({e}). Falling back to CSVLogger.")
            tb_logger = CSVLogger(save_dir=log_dir, name="tft_training")

        # Safe hardware detection using PyTorch directly
        if use_gpu and torch.cuda.is_available():
            accelerator = "gpu"
            devices = 1
        elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = "auto"

        logger.info(f"Configuring Trainer. Selected Accelerator: {accelerator.upper()}")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            enable_model_summary=True,
            gradient_clip_val=0.1,  # Prevents exploding gradients
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=tb_logger,
        )

        return trainer