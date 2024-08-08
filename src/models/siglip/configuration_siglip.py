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
