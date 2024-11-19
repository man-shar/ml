from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig

import torch

from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    print("read files")
    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device, dtype=torch.float32)

    # Load the state dict of the model
    keys_not_found = model.load_state_dict(tensors, strict=False)

    if len(keys_not_found):
        print(
            f"Keys not found in your model, but were present in state dict: {keys_not_found}"
        )

    state_dict = model.state_dict()
    num_parameters = sum(p.numel() for p in model.parameters())
    num_state_dict = sum(p.numel() for p in state_dict.values())
    print(
        "num parameters = {}, stored in state_dict = {}, diff = {}".format(
            num_parameters, num_state_dict, num_state_dict - num_parameters
        )
    )

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
