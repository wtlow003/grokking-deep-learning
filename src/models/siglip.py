from typing import Optional, Tuple

import torch
import torch.nn as nn


class SiglipVisionConfig:
    """
    Configuration for Siglip Vision Model
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Siglip Vision Model Configuration

        Args:
            hidden_size (int, optional): Dimension of encoder and pooler layers. Defaults to 768.
            intermediate_size (int, optional): Dimension of feed-forward layer in Transformers encoder. Defaults to 3072.
            num_hidden_layers (int, optional): Number of hidden layers in the Transformer encoder. Defaults to 12.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 12.
            num_channels (int, optional): Number of channels in input images. Defaults to 3.
            image_size (int, optional): Size of input images. Defaults to 224.
            patch_size (int, optional): Patch size. Defaults to 16.
            layer_norm_eps (float, optional): Layer norm epsilon. Defaults to 1e-6.
            attention_dropout (float, optional): Attention dropout. Defaults to 0.0.
            num_image_tokens (int, optional): Number of image tokens. Defaults to None.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


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


class SiglipVisionTransformer(nn.Module):
    """
    Siglip Vision Transformer
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
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
