from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_siglip import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    """
    Siglip Vision Embeddings
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # no padding is added
        )

        # This line calculates the total number of patches in the image.
        # It does this by:
        # 1. Dividing the image size by the patch size (integer division)
        #    to get the number of patches along one dimension.
        # 2. Squaring the result to get the total number of patches,
        #    since the image is assumed to be square.
        # For example, if image_size is 224 and patch_size is 16,
        # we get (224 // 16) ** 2 = 14 ** 2 = 196 patches.
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand(1, -1),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Forward pass of Siglip Vision Embeddings

        Args:
            pixel_values (torch.Tensor): Pixel values of input images.

        Returns:
            torch.Tensor: Embeddings of input images.
        """
        # [batch_size, num_channels, height, width]
        _, _, height, width = pixel_values.shape
        assert height == width, "Image height and width must be equal"
        assert (
            height == self.image_size
        ), "Image height and width must be equal to image_size"
        assert (
            width == self.image_size
        ), "Image height and width must be equal to image_size"

        # convolve the pixel values with the patch embedding -> no padding is added
        # [batch_size, num_channels, height, width] -> [batch_size, embedding_dim, num_patches_H, num_patches_W]
        patch_embeds = self.patch_embedding(pixel_values)
        # [batch_size, embedding_dim, num_patches_H, num_patches_W] -> [batch_size, embedding_dim, num_patches]
        # where num_patches = num_patches_H * num_patches_W
        embeddings = patch_embeds.flatten(2)
        # [batch_size, embedding_dim, num_patches] -> [batch_size, num_patches, embedding_dim]
        embeddings = embeddings.transpose(1, 2)
        # add position embeddings to each patch, where positional encoding is vector of size [embedding_dim]
        embeddings += self.position_embedding(self.position_ids)
        # [batch_size, num_patches, embedding_dim]
        return embeddings


class SiglipAttention(nn.Module):
    """
    Siglip Attention – "Multi-headed attention from 'Attention is All You Need'"
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5  # equivalent to 1 / sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Siglip Attention

        Args:
            hidden_states (torch.Tensor): Hidden states of the last layer.

        Returns:
            torch.Tensor: Hidden states of the last layer.
        """
        # [batch_size, num_patches, embedding_dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [batch_size, num_patches, embedding_dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [batch_size, num_patches, embedding_dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [batch_size, num_patches, embedding_dim]
        value_states = self.v_proj(hidden_states)

        # we are splitting embeddings into heads: [embedding_dim] -> [num_heads, head_dim], e.g, 1024 -> 8, 128
        # query_states: [batch_size, num_patches, num_heads, head_dim] transposed to [batch_size, num_heads, num_patches, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # key_states: [batch_size, num_heads, num_patches, head_dim]
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # value_states: [batch_size, num_heads, num_patches, head_dim]
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # compute attention scores ('affinities') between query and key states
        # [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should have shape {(batch_size, self.num_heads, seq_len, seq_len)}, but got {attn_weights.size()}"
            )

        # apply softmax row-wise
        # [batch_size, num_heads, num_patches, num_patches] -> [batch_size, num_heads, num_patches, num_patches]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        # apply dropout to attention weights
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        # multiply attention weights with value states
        # [batch_size, num_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should have shape {(batch_size, self.num_heads, seq_len, self.head_dim)}, but got {attn_output.size()}"
            )

        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embedding_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [batch_size, num_patches, embedding_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    """
    Siglip MLP
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of Siglip MLP

        Args:
            hidden_states (torch.Tensor): Hidden states of the last layer.

        Returns:
            torch.Tensor: Hidden states of the last layer.
        """
        # [batch_size, num_patches, embedding_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # apply activation (non-linear) function
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embedding_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipVisionEncoderLayer(nn.Module):
    """
    Siglip Vision Encoder Layer
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of Siglip Vision Encoder Layer

        Args:
            hidden_states (torch.Tensor): Hidden states of the last layer.

        Returns:
            torch.Tensor: Hidden states of the last layer.
        """
        # [batch_size, num_patches, embedding_dim]
        residual = hidden_states
        # [batch_size, num_patches, embedding_dim] -> [batch_size, num_patches, embedding_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embedding_dim] -> [batch_size, num_patches, embedding_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch_size, num_patches, embedding_dim]
        hidden_states = residual + hidden_states
        residual = hidden_states
        # [batch_size, num_patches, embedding_dim] -> [batch_size, num_patches, embedding_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embedding_dim] -> [batch_size, num_patches, embedding_dim]
        hidden_states = self.mlp(hidden_states)
        # [batch_size, num_patches, embedding_dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipVisionEncoder(nn.Module):
    """
    Siglip Vision Encoder
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass of Siglip Vision Encoder

        Args:
            input_embeds (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Hidden states of the last layer.
        """
        # [batch_size, num_patches, embedding_dim]
        hidden_states = input_embeds

        for encoder_layer in self.layers:
            # [batch_size, num_patches, embedding_dim] -> [batch_size, num_patches, embedding_dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    Siglip Vision Transformer
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        # docs: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass of Siglip Vision Transformer

        Args:
            pixel_values (torch.Tensor): Pixel values of input images.

        Returns:
            torch.Tensor: Hidden states of the last layer.
        """
        # [batch_size, num_channels, height, width] -> [batch_size, num_patches, embedding_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """
    Siglip Vision Model
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple:
        """Forward pass of Siglip Vision Model

        Args:
            pixel_values (torch.Tensor): Pixel values of input images.

        Returns:
            Tuple: _description_
        """
        # [batch_size, num_channels, height, width] -> [batch_size, num_patches, embedding_dim]
        return self.vision_model(pixel_values=pixel_values)
