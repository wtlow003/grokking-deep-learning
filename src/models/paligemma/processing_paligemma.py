from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(
    prefix_prompt: str,
    bos_token: str,
    image_seq_len: int,
    image_token: str,
) -> str:
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """
    Resize the image to the given size.

    Args:
        image (Image.Image): The image to resize.
        size (Tuple[int, int]): The size to resize the image to.
        resample (Image.Resampling): The resampling filter to use.
        reducing_gap (Optional[int]): The reducing gap to use.

    Returns:
        Image.Image: The resized image.
    """
    resized_image = image.resize(size, resample, reducing_gap=reducing_gap)
    return resized_image


def rescale(
    image: np.ndarray, scale: float, dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    Rescale the image to the given scale.

    Args:
        image (np.ndarray): The image to rescale.
        scale (float): The scale to rescale the image to.
        dtype (Optional[np.dtype]): The dtype to use.

    Returns:
        np.ndarray: The rescaled image.
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def normalize(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """
    Normalize the image to the given mean and std.

    Args:
        image (np.ndarray): The image to normalize.
        mean (Union[float, List[float]]): The mean to normalize the image to.
        std (Union[float, List[float]]): The std to normalize the image to.

    Returns:
        np.ndarray: The normalized image.
    """
    mean = np.array(mean, dtype=image.dtype)  # type: ignore
    std = np.array(std, dtype=image.dtype)  # type: ignore
    normalized_image = (image - mean) / std
    return normalized_image


def process_images(
    images: List[Image.Image],  # type: ignore
    size: Tuple[int, int],
    resample: Image.Resampling,
    rescale_factor: float,
    image_mean: List[float],
    image_std: List[float],
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]  # type: ignore
    # convert each image to a numpy array
    images: List[np.ndarray] = [np.array(image) for image in images]
    # rescale pixel values to be in range [0, 1]
    images = [
        rescale(image, scale=rescale_factor, dtype=np.float32) for image in images  # type: ignore
    ]
    # normalize the images
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # move channel dimension to the first dimension
    # expected input shape: (num_channels, height, width)
    images = [image.transpose(2, 0, 1) for image in images]

    return images


class PaliGemmaProcessor:
    """
    Input processor for PaliGemma model.
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initialize the processor.

        Args:
            tokenizer (_type_): _description_
            num_image_tokens (int): _description_
            image_size (int): _description_
        """
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # tokenizer setup: https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma
        #
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        # tokens for object detection (bounding boxes)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        # tokens for object segmentation
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # add BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        assert (
            len(images) == 1 and len(text) == 1
        ), f"Received {len(images)} images and {len(text)} texts, but the PaliGemma model only supports one image and one text."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # convert the list of numpy arrays to a single numpy array
        # [batch_size, num_channels, height, width]
        pixel_values = np.stack(pixel_values, axis=0)
        # convert numpy array to torch tensor
        pixel_values = torch.tensor(pixel_values)

        # prepend a `self.image_seq_length` number of images tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # return input_ids and attention_mask as pytorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
