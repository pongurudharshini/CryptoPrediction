"""
Temporal Fusion Transformer (TFT) & Spatio-Temporal Graph Architecture Builder.
Constructs:
1. MultimodalCrossAttentionFusion: Fuses LLM/FinBERT semantic embeddings with technical data.
2. CrossCurrencyGraphTransformer: Spatio-Temporal Graph Neural Network (ST-GNN) modeling 
   cross-currency spillovers and dynamic inter-asset dependency matrices (Singh & Bhat, 2024).
3. TFTBuilder: Initializes TFT architectures with Quantile Loss for probabilistic risk bounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import TimeSeriesDataSet
import logging
from models.custom_losses import AsymmetricLinexLoss

logger = logging.getLogger(__name__)

class MultimodalCrossAttentionFusion(nn.Module):
    """
    Academic Novelty: Cross-Attention Mechanism for Multimodal LLM Sentiment.
    Fuses dense vector embeddings from FinBERT (Text/Socials/TikTok) with numerical 
    pricing features extracted by the Temporal Fusion Transformer.
    """
    def __init__(self, tft_hidden_size: int, llm_embedding_size: int = 768, num_heads: int = 4):
        super(MultimodalCrossAttentionFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=tft_hidden_size, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.llm_proj = nn.Linear(llm_embedding_size, tft_hidden_size)
        self.layer_norm = nn.LayerNorm(tft_hidden_size)
        
    def forward(self, tft_features: torch.Tensor, llm_embeddings: torch.Tensor) -> torch.Tensor:
        llm_proj = self.llm_proj(llm_embeddings)
        attn_output, _ = self.cross_attention(
            query=tft_features,
            key=llm_proj,
            value=llm_proj
        )
        fused_output = self.layer_norm(tft_features + attn_output)
        return fused_output


class CrossCurrencyGraphTransformer(nn.Module):
    """
    Academic Novelty: Spatio-Temporal Graph Transformer for Cross-Currency Market Spillover.
    Models 10 cryptocurrencies as interconnected graph nodes to capture systemic contagion
    and cross-asset lead-lag effects (Singh & Bhat, 2024).
    """
    def __init__(self, num_nodes: int = 10, in_features: int = 16, hidden_dim: int = 32, num_heads: int = 4):
        super(CrossCurrencyGraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Spatial Graph Attention: Learns dynamic inter-asset adjacency matrices
        self.spatial_q = nn.Linear(in_features, hidden_dim)
        self.spatial_k = nn.Linear(in_features, hidden_dim)
        self.spatial_v = nn.Linear(in_features, hidden_dim)
        
        # Temporal Sequence Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection back to feature dimension
        self.out_proj = nn.Linear(hidden_dim, in_features)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, node_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: Tensor of shape (batch_size, num_nodes, seq_len, in_features)
        Returns:
            fused_features: Spatio-temporally contextualized features (batch_size, num_nodes, seq_len, in_features)
            adj_matrix: Dynamic inter-asset correlation attention weights (batch_size, num_nodes, num_nodes)
        """
        B, N, T, F_in = node_features.shape
        
        # Aggregate temporal sequence per node for spatial correlation estimation
        node_summary = node_features.mean(dim=2)  # Shape: (B, N, F_in)
        
        # Calculate dynamic Spatial Adjacency Matrix W_t using Scaled Dot-Product Attention
        Q = self.spatial_q(node_summary)  # (B, N, hidden_dim)
        K = self.spatial_k(node_summary)  # (B, N, hidden_dim)
        V = self.spatial_v(node_summary)  # (B, N, hidden_dim)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        adj_matrix = F.softmax(scores, dim=-1)  # Dynamic cross-currency spillover graph (B, N, N)
        
        # Spatial Message Passing: Aggregate features across correlated assets
        spatial_context = torch.bmm(adj_matrix, V)  # (B, N, hidden_dim)
        
        # Reshape to inject spatial context across temporal sequences
        spatial_expanded = spatial_context.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, hidden_dim)
        
        # Temporal sequence processing per node
        spatial_flat = spatial_expanded.view(B * N, T, self.hidden_dim)
        temporal_out = self.temporal_transformer(spatial_flat)
        temporal_reshaped = temporal_out.view(B, N, T, self.hidden_dim)
        
        # Output projection and residual connection
        projected = self.out_proj(temporal_reshaped)
        fused_features = self.norm(node_features + projected)
        
        return fused_features, adj_matrix


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
        """
        logger.info("Initializing Temporal Fusion Transformer Architecture...")
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=loss_fn,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"TFT Model built successfully with {total_params:,} trainable parameters.")
        return model