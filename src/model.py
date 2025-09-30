"""
Neural network models for HCP fMRI time series prediction.
Modified to store all configuration parameters as instance variables.
"""

import numpy as np
import torch
import torch.nn as nn


class CustomAttention(nn.Module):
    """Attention mechanism with shared K/V and source-target aware queries"""

    def __init__(self, dim, bottleneck_dim, num_regions, attention_dropout_rate):
        super(CustomAttention, self).__init__()
        assert dim > 0, f"Dimension must be positive, got {dim}"
        assert bottleneck_dim > 0, f"Bottleneck dimension must be positive, got {bottleneck_dim}"
        assert bottleneck_dim <= dim, f"Bottleneck dimension ({bottleneck_dim}) must be <= dim ({dim})"
        assert num_regions > 0, f"Number of regions must be positive, got {num_regions}"
        assert 0 <= attention_dropout_rate <= 1, f"Dropout rate must be in [0,1], got {attention_dropout_rate}"

        self.num_regions = num_regions
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim

        # Shared projections for keys and values
        self.shared_key_proj = nn.Linear(dim, dim)
        self.shared_value_proj = nn.Linear(dim, bottleneck_dim)

        # Source-target specific query projections as a parameter tensor
        # Shape: (num_regions_source, num_regions_target, input_dim, input_dim)
        self.query_weight = nn.Parameter(torch.randn(num_regions, num_regions, dim, dim))
        self.query_bias = nn.Parameter(torch.randn(num_regions, num_regions, dim))

        # Attention dropout for regularization
        self.attention_dropout = nn.Dropout(attention_dropout_rate)

        self.final_value_reproj = nn.Linear(bottleneck_dim, dim)

    def forward(self, q_embed, kv_embed):
        """
        Args:
            q_embed: (batch_size, num_regions, dim) - region-specific embeddings for queries
            kv_embed: (batch_size, num_regions, dim) - shared embeddings for keys and values

        Returns:
            attended_outputs: (batch_size, num_regions, dim)
            attention_weights: (batch_size, num_regions, num_regions)
        """
        batch_size, num_regions_q, dim_q = q_embed.shape
        batch_size_kv, num_regions_kv, dim_kv = kv_embed.shape

        assert batch_size == batch_size_kv, f"Batch size mismatch: q={batch_size}, kv={batch_size_kv}"
        assert num_regions_q == self.num_regions, (
            f"Query regions mismatch: expected {self.num_regions}, got {num_regions_q}"
        )
        assert num_regions_kv == self.num_regions, (
            f"Key/value regions mismatch: expected {self.num_regions}, got {num_regions_kv}"
        )
        assert dim_q == self.dim, f"Query dimension mismatch: expected {self.dim}, got {dim_q}"
        assert dim_kv == self.dim, f"Key/value dimension mismatch: expected {self.dim}, got {dim_kv}"

        # Compute shared keys and values for all regions
        K = self.shared_key_proj(kv_embed)  # (batch_size, num_regions, dim)
        V = self.shared_value_proj(kv_embed)  # (batch_size, num_regions, bottleneck_dim)

        V_normalized = nn.functional.normalize(V, dim=-1)  # Normalize to keep the effect of all values the same

        # Compute source-target specific queries vectorized
        # Q[i,j] = query from source region i to target region j
        # Using einsum for efficient computation: (batch, source, dim) * (source, target, dim, dim) -> (batch, source, target, dim)
        Q = torch.einsum("bsi,stid->bstd", q_embed, self.query_weight) + self.query_bias.unsqueeze(
            0
        )  # unsqueeze to account for batch dim

        # Compute attention scores vectorized: Q[i,j] · K[j] for each source i and target j
        # Using einsum: (batch, source, target, dim) * (batch, target, dim) -> (batch, source, target)
        attention_weights = torch.einsum("bstd,btd->bst", Q, K)

        # Only keep positive weights and normalize attention weights for each source region to sum to 1.0
        attention_weights = attention_weights.relu()
        attention_weights_sum = 1e-8 + attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = attention_weights / attention_weights_sum

        # Apply dropout to attention to nudge model to try to use a variety of other regions
        attention_weights = self.attention_dropout(attention_weights)

        # Compute attended outputs: weighted sum of values
        attended_outputs = torch.bmm(attention_weights, V_normalized)
        attended_outputs = self.final_value_reproj(attended_outputs)

        return attended_outputs, attention_weights


class HCPAttentionModel(nn.Module):
    """Time series prediction model using symmetric attention mechanism"""

    def __init__(
        self,
        num_regions,
        input_window,
        output_window,
        hidden_dim,
        num_input_layers,
        num_prediction_layers,
        bottleneck_dim,
        attention_dropout_rate,
        mean,
        stddev,
    ):
        super(HCPAttentionModel, self).__init__()

        # Validate all input parameters
        assert num_regions > 0, f"Number of regions must be positive, got {num_regions}"
        assert input_window > 0, f"Input window must be positive, got {input_window}"
        assert output_window > 0, f"Output window must be positive, got {output_window}"
        assert hidden_dim > 0, f"Hidden dimension must be positive, got {hidden_dim}"
        assert num_input_layers > 0, f"Number of input layers must be positive, got {num_input_layers}"
        assert num_prediction_layers > 0, f"Number of prediction layers must be positive, got {num_prediction_layers}"
        assert bottleneck_dim > 0, f"Bottleneck dimension must be positive, got {bottleneck_dim}"
        assert bottleneck_dim <= hidden_dim, (
            f"Bottleneck dimension ({bottleneck_dim}) must be <= hidden dimension ({hidden_dim})"
        )
        assert 0 <= attention_dropout_rate <= 1, (
            f"Attention dropout rate must be in [0,1], got {attention_dropout_rate}"
        )
        assert len(mean) == num_regions, f"Mean array length ({len(mean)}) must match num_regions ({num_regions})"
        assert len(stddev) == num_regions, f"Stddev array length ({len(stddev)}) must match num_regions ({num_regions})"
        assert np.all(np.array(stddev) > 0), f"All standard deviations must be positive, got min: {np.min(stddev)}"

        # Store all configuration parameters as instance variables
        self.num_regions = num_regions
        self.input_window = input_window
        self.output_window = output_window
        self.hidden_dim = hidden_dim
        self.num_input_layers = num_input_layers
        self.num_prediction_layers = num_prediction_layers
        self.bottleneck_dim = bottleneck_dim
        self.attention_dropout_rate = attention_dropout_rate

        # Store normalization parameters as buffers in reshaped form for broadcasting
        # Shape: (1, num_regions, 1) for broadcasting with (batch_size, num_regions, input/output_window)
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1)
        stddev_tensor = torch.tensor(stddev, dtype=torch.float32).view(1, -1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("stddev", stddev_tensor)

        # Shared embedding for keys and values (same for all regions)
        self.shared_kv_input_projection = nn.Linear(input_window, hidden_dim)
        self.shared_kv_blocks = self._build_layers(hidden_dim, num_input_layers - 1)

        # Per region input embedding layers for the query
        self.query_input_projection = nn.ModuleList([nn.Linear(input_window, hidden_dim) for _ in range(num_regions)])
        self.query_blocks = nn.ModuleList(
            [self._build_layers(hidden_dim, num_input_layers - 1) for _ in range(num_regions)]
        )

        # Per region output heads after the attention
        self.post_attention_blocks = nn.ModuleList(
            [self._build_layers(hidden_dim, num_prediction_layers - 1) for _ in range(num_regions)]
        )
        self.output_projection_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, output_window) for _ in range(num_regions)]
        )

        # Custom attention module
        self.attention_module = CustomAttention(hidden_dim, bottleneck_dim, num_regions, attention_dropout_rate)

    def _build_layers(self, dim, num_layers):
        num_layers = max(0, num_layers)
        layers = []
        for i in range(num_layers):
            layers.extend([nn.ReLU(), nn.Linear(dim, dim)])

        return nn.Sequential(*layers)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, num_regions, input_window)
            return_attention: Whether to return attention weights

        Returns:
            predictions: (batch_size, num_regions, output_window)
            attention_weights: (batch_size, num_regions, num_regions) if return_attention=True
        """
        batch_size, num_regions, input_window = x.shape
        assert num_regions == self.num_regions, (
            f"Input regions ({num_regions}) must match model regions ({self.num_regions})"
        )
        assert input_window == self.input_window, (
            f"Input window ({input_window}) must match model input window ({self.input_window})"
        )
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"

        # Apply per-region z-normalization to input
        # x shape: (batch_size, num_regions, input_window)
        # mean and stddev shape: (1, num_regions, 1) - already reshaped for broadcasting
        x_normalized = (x - self.mean) / (self.stddev + 1e-8)

        # Initial projection and query-specific processing for all regions
        query_embed = [
            self.query_blocks[i](self.query_input_projection[i](x_normalized[:, i, :])) for i in range(self.num_regions)
        ]
        query_embed = torch.stack(query_embed, dim=1)

        # Compute shared key-value embeddings (optimized batch processing)
        x_reshaped = x_normalized.view(batch_size * self.num_regions, self.input_window)
        kv_projected = self.shared_kv_input_projection(x_reshaped)
        key_value_embed = self.shared_kv_blocks(kv_projected).view(batch_size, self.num_regions, self.hidden_dim)

        # Apply attention: region-specific queries, shared keys and values
        attended_outputs, attention_weights = self.attention_module(q_embed=query_embed, kv_embed=key_value_embed)

        # Post-attention processing and output projection for all regions
        region_pred = [
            self.output_projection_layers[i](self.post_attention_blocks[i](attended_outputs[:, i, :]))
            for i in range(self.num_regions)
        ]
        region_pred = torch.stack(region_pred, dim=1)

        # Un-normalize predictions (undo the z-normalization)
        # region_pred shape: (batch_size, num_regions, output_window)
        # mean and stddev shape: (1, num_regions, 1) - already reshaped for broadcasting
        region_pred_unnormalized = region_pred * self.stddev + self.mean

        assert list(attention_weights.shape) == [batch_size, self.num_regions, self.num_regions], (
            f"Unexpected attention weights shape: {attention_weights.shape}"
        )
        assert list(region_pred_unnormalized.shape) == [
            batch_size,
            self.num_regions,
            self.output_window,
        ], f"Unexpected final output shape: {region_pred_unnormalized.shape}"

        if return_attention:
            return region_pred_unnormalized, attention_weights

        return region_pred_unnormalized
