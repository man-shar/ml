from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import LlamaTokenizer


# Details here: https://huggingface.co/blog/paligemma#detailed-inference-process

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(
    prefix_prompt: str, bos_token: int, image_seq_len: int, image_token: str
):
    # https://huggingface.co/blog/paligemma#detailed-inference-process
    # Detailed Inference Process
    # The input text is tokenized normally. A <bos> token is added at the beginning, and an additional newline token (\n) is appended. This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    # in the paper, they say that they process "\n" separately, but hf does it this way
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Optional[Union[float, List[float]]] = None,
    std: Optional[Union[float, List[float]]] = None,
):
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)

    # normalized_image = image
    # normalized_image = normalized_image - mean
    # normalized_image = normalized_image / std

    # return normalized_image
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # convert each image to np array
    images = [np.array(image) for image in images]

    # rescale the pixel values to be in range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]

    # Normalise the images to have mean image_mean and std dev image_std
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    # move the channel dimension to the first dimension
    # model expect [channel, height, width]
    images = [image.transpose(2, 0, 1) for image in images]

    return images


class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(
        self, tokenizer: LlamaTokenizer, num_image_tokens: int, image_size: int
    ) -> None:
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # create additional tokens for image
        # check the readme here in the Tokenizer section
        # https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # 1024 tokens: <loc0000>...<loc1023> for image segmentation
        # 128 tokens: <seg000>...<seg127> for object detection
        SEGMENTATION_TOKENS = [f"<loc{str(i).zfill(4)}>" for i in range(0, 1024)]

        OBJECT_DETECTION_TOKENS = [f"<seg{str(i).zfill(3)}>" for i in range(0, 128)]

        EXTRA_TOKENS = SEGMENTATION_TOKENS + OBJECT_DETECTION_TOKENS

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # BOS and EOS tokens
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
        ), f"Received {len(images)} images for {len(text)} prompts."

        pixel_values: List[torch.Tensor] = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # convert list of tensors to one tensor with [batch_size, channel, h, w]
        pixel_values = np.stack(pixel_values, axis=0)

        # Convert the numpy array to pytorch tensor
        # becomes [batch, channel, h, w]
        pixel_values = torch.tensor(pixel_values)

        # prepend a `self.image_seq_length` number of image tokens to the prompt

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings, return_tensors="pt", padding=padding, truncation=truncation
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
