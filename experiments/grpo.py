import gc
import time
import torch
from vllm import LLM, LLMEngine, SamplingParams, EngineArgs, RequestOutput
from vllm.sequence import PromptLogprobs, SampleLogprobs
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from pydantic import BaseModel
from typing import Optional, Literal, Callable
import wandb
from tqdm import tqdm
import numpy as np
from transformers import Qwen3ForCausalLM

from torch.optim import Optimizer
import os

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def optimizer_state_size_mb(optimizer: Optimizer) -> str:
    """
    Returns a string with the size of an optimizer in GB
    """

    def _bytes(obj) -> int:
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple, set)):
            return sum(_bytes(v) for v in obj)
        return 0

    total_bytes = 0
    for state in optimizer.state.values():
        total_bytes += _bytes(state)
    return f"{total_bytes / (1024**3):.4f}GB"


def gpu_mem_allocated(prefix=""):
    """
    Logs GPU memory usage.
    """
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    print(f"{prefix} allocated={a:.1f}GB reserved={r:.1f}GB")
    # print(torch.cuda.memory_stats())


def clear_gpu():
    """
    Hacky way to clear gpu. Doesn't do anything when I actually need it to.
    """
    for r in range(3):
        # Force cleanup
        gc.collect()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    with torch.no_grad():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    gpu_mem_allocated()


model_name = "Qwen/Qwen3-1.7B"
dataset_name = "trl-lib/tldr"

# stolen from grpo_trainer.py in HF. VLLM needs these in `external_launcher` mode
os.environ["RANK"] = str(0)
os.environ["LOCAL_RANK"] = str(0)
os.environ["WORLD_SIZE"] = str(1)
os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12345")

# Use lower GPU memory for vLLM to leave room for training model
engine_args = EngineArgs(
    model=model_name,
    gpu_memory_utilization=0.15,  # Match the working script
    dtype="bfloat16",
    max_model_len=1024,  # Limit context to save memory
    max_num_batched_tokens=4096,
    max_num_seqs=8,  # to prevent VLLM hogging
    # Stolen from hf
    # external launcher seems to start vllm in the same process as this file. If we run without this, vllm starts its own process. Perhaps memory saving?
    distributed_executor_backend="external_launcher",
    # Feed identical seed for tp groups to ensure sampling results are the same across workers
    seed=32,
)

vllm = LLM(**vars(engine_args))


tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)

# This is the model we will RL on
train_model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},  # Explicitly use GPU 0 instead of "auto"
    use_cache=False,
)

# Enable gradient checkpointing to save memory
train_model.gradient_checkpointing_enable()

dataset = load_dataset(dataset_name)

pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id


def update_vllm_weights(self):
    """
    Updates the model weights in vllm with our trained model
    """
    for name, p in train_model.named_parameters():
        llm_model = self.model_runner.model
        llm_model.load_weights([(name, p.data)])


def extract_chosen_logprobs(
    token_ids: list[int], vllm_logprobs: SampleLogprobs | PromptLogprobs
):
    """
    Extracts the correct logprobs for a given token from all logits/logprobs
    usually logits are [batch, seq, d_vocab]
    when we already know the completion tokens, then we know which logit to pick from the d_vocab logits
    does what torch.gather would do (see gather usage later in the train function)
    """
    return torch.tensor(
        [lp[token_id].logprob for token_id, lp in zip(token_ids, vllm_logprobs)],
        device="cpu",
        dtype=torch.bfloat16,
    )


def rollout(
    prompts: str | list[str], sampling_params: SamplingParams, debug: bool = False
):
    """
    Generate data using the vllm loaded model
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend(
            [prompt] * n_samples
        )  # Repeat each prompt n_samples times

    # only generate 1  sample because of the repeat we did above
    sampling_params.n = 1
    generations: list[RequestOutput] = vllm.generate(expanded_prompts, sampling_params)

    groups = []
    for i in range(0, len(generations), n_samples):
        output = generations[i]
        group = {
            "input_string": output.prompt,
            "input_tokens": output.prompt_token_ids,
            # first token has no logprobs
            # (note that vllm internally reorders logprobs such that the token at index i has its OWN logprobs, and NOT for the next token like we would normally expect from LLM outputs)
            "input_logprobs": extract_chosen_logprobs(
                output.prompt_token_ids[1:], output.prompt_logprobs[1:]
            ),
            "samples": [],
        }

        for j in range(n_samples):
            output = generations[i + j].outputs[0]  # Only 1 output per prompt now
            group["samples"].append(
                {
                    "completion_string": output.text,
                    "completion_tokens": output.token_ids,
                    "non_padded_full_length": len(group["input_tokens"])
                    + len(output.token_ids),
                    "non_padded_completion_length": len(output.token_ids),
                    "old_logprobs": extract_chosen_logprobs(
                        output.token_ids, output.logprobs
                    ),
                }
            )

        groups.append(group)

    for i, g in enumerate(groups):
        # input lengths
        g["max_full_length"] = max(
            (len(s["completion_tokens"]) + len(g["input_tokens"])) for s in g["samples"]
        )
        g["max_completion_length"] = max(
            len(s["completion_tokens"]) for s in g["samples"]
        )

        # pad this group's input ids to the left by max_prompt_length - len(g["input_tokens"])
        g["input_padding_length"] = max_prompt_length - len(g["input_tokens"])
        g["non_padded_input_length"] = len(g["input_tokens"])

        assert g["input_padding_length"] >= 0, (
            f"Invalid inputs for group: {i}, input length: {len(g['input_tokens'])}"
        )

        if g["input_padding_length"] > 0:
            if debug:
                print(
                    f"Padded group {i} with input length {len(g['input_tokens'])} by {g['input_padding_length']}"
                )
            g["input_tokens"] = [pad_token_id] * g["input_padding_length"] + g[
                "input_tokens"
            ]

        else:
            g["input_padding_length"] = 0

        if debug:
            print(f"Group {i}")
            print(f"-- Non padded input length: {g['non_padded_input_length']}")
            print(f"-- Maximum full length: {g['max_full_length']}")
            print(f"-- Maximum completion length: {g['max_completion_length']}")

        # pad the completion ids to the right to match the max_completion_length
        for j, s in enumerate(g["samples"]):
            # input lengths are hte same for all samples in a group
            s["completion_padding_length"] = max_completion_length - len(
                s["completion_tokens"]
            )

            if s["completion_padding_length"] > 0:
                if debug:
                    print(
                        f"-- Padded sample {j} with completion length {len(s['completion_tokens'])} by {s['completion_padding_length']}"
                    )

                # pad the completion tokens
                s["completion_tokens"] = (
                    s["completion_tokens"]
                    + [pad_token_id] * s["completion_padding_length"]
                )
            else:
                s["completion_padding_length"] = 0

        # create attention masks for this group
        g["attention_mask"] = [
            # first the input padding tokens get 0
            [0] * g["input_padding_length"]
            +
            # input tokens get 1 (we padded this above so this is a little annoying)
            [1] * (len(g["input_tokens"]) - g["input_padding_length"])
            +
            # completion tokens get 1. again padded above so annoying and have to subtract
            [1] * (len(s["completion_tokens"]) - s["completion_padding_length"])
            +
            # completion padding tokens get 0
            [0] * s["completion_padding_length"]
            for s in g["samples"]
        ]

    return groups


LossMethod = Callable[[dict, np.ndarray], torch.Tensor]

clip_eps = 0.1

tiny = 1e-5


def grpo(group: dict, new_logprobs: torch.Tensor, rewards: np.ndarray):
    """
    Returns the loss computation using GRPO
    """
    # normalise all rewards within this group of samples
    advantages = (rewards - rewards.mean()) / (rewards.std() + tiny)
    advantages = torch.tensor(advantages, device=train_model.device, dtype=torch.float)

    loss = torch.tensor(0.0, device=train_model.device)

    for i, sample in enumerate(group["samples"]):
        adv = advantages[i]

        # Detach old_logprobs to prevent keeping old graph in memory
        importance_ratios = torch.exp(
            new_logprobs[i, : sample["non_padded_completion_length"]]
            - sample["old_logprobs"].to(device=train_model.device)
        )

        # clip ratios
        clipped_importance_ratios = torch.clip(
            importance_ratios, 1 - clip_eps, 1 + clip_eps
        )

        sample_loss = torch.minimum(
            input=importance_ratios * adv, other=clipped_importance_ratios * adv
        ).mean()

        loss += sample_loss

    loss /= len(group["samples"])

    return -loss


def simple_reward(sample: dict):
    return -len(sample["completion_string"])


step = 0
epochs = 1
n_samples = 4
lr = 1e-6
batch_size = 1
gradient_accumulation_steps = 1
max_prompt_length = 512
max_completion_length = 512


template_tokenized = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": "Your role is to summarize the text into a concise summary. Do not think excessively, focus on summarizing the text.",
        },
        {"role": "user", "content": "The text you must summarize is:\n"},
    ],
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True,
)


def convert_prompt_to_chat_format(batch: dict) -> list[dict[str, str]]:
    """
    Convert a prompt to a chat format.
    """

    # do stupid things just to clip the correct part of the prompt (we cliip the main message, not the special tokens or the formatting)
    prompts_tokenized = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=(max_prompt_length - len(template_tokenized)),
    )["input_ids"]

    prompts_clipped_decoded = tokenizer.batch_decode(prompts_tokenized)

    return {
        "prompt": tokenizer.apply_chat_template(
            [
                [
                    {
                        "role": "system",
                        "content": "Your role is to summarize the text into a concise summary. Do not think excessively, focus on summarizing the text.",
                    },
                    {
                        "role": "user",
                        "content": "The text you must summarize is:\n"
                        + p.rstrip("TL;DR:").strip(),
                    },
                ]
                for p in prompts_clipped_decoded
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    }


ds = dataset["train"].map(
    convert_prompt_to_chat_format, remove_columns="completion", batched=True
)

train_loader = DataLoader(ds, batch_size=batch_size)


def process_group(g: dict, g_idx, loss_fn: Callable):
    """
    Processes a single group.
    """
    print(
        f"Processing group with input length: {len(g['input_tokens'])} and max completion length: {g['max_completion_length']}"
    )

    if g["max_completion_length"] < 100:
        print(g["input_string"])
        print([s["completion_string"] for s in g])

    rewards = [simple_reward(s) for s in g["samples"]]

    # put full input's tokens
    group_input_ids = torch.tensor(
        [g["input_tokens"] + s["completion_tokens"] for s in g["samples"]],
        dtype=torch.long,
        device=train_model.device,
    )

    # print("input_ids.shape", group_input_ids.shape)

    # put attention masks on the gpu
    attention_mask = torch.tensor(
        g["attention_mask"],
        dtype=torch.long,
        device=train_model.device,
    )

    # print("attention_mask.shape", attention_mask.shape)

    assert group_input_ids.shape[1] == max_prompt_length + max_completion_length, (
        f"Expected: {max_prompt_length + max_completion_length}, found: {group_input_ids.shape}"
    )

    assert attention_mask.shape[1] == max_prompt_length + max_completion_length, (
        f"Expected: {max_prompt_length + max_completion_length}, found: {attention_mask.shape}"
    )

    gpu_mem_allocated("before model forward")

    # keep only completion logits
    logits_to_keep = max_completion_length + 1

    outputs = train_model(
        input_ids=group_input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep,
        use_cache=False,
    )

    gpu_mem_allocated("after model forward")
    logits: torch.Tensor = outputs.logits

    # print("logits shape", logits.shape)

    # remove the last token's logits
    logits = logits[:, :-1, :]

    # prepare index to gather all the completion token's logits
    index = group_input_ids[:, -max_completion_length:].unsqueeze(-1)

    # print("index shape", index.shape)

    # gather required logits
    selected_logits = torch.gather(
        logits,
        dim=-1,
        index=index,
    ).squeeze(-1)

    logsumexp = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])

    # print("selected_logits.shape", selected_logits.shape)
    # print("logsumexp.shape", logsumexp.shape)

    new_logprobs = selected_logits - logsumexp

    # Compute loss for this group
    loss = loss_fn(g, new_logprobs, np.array(rewards))

    # Scale for gradient accumulation and backward immediately
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    loss_item = loss.item()

    del group_input_ids, logits, outputs, attention_mask, loss, scaled_loss

    return sum(rewards), loss_item


def train(loss_method: Literal["grpo", "gspo"] = "grpo"):
    # wandb.init(
    #     project="grpo",
    #     name=f"grpo-{time.strftime('%Y-%m-%d--%H-%M')}",
    #     config={"epochs": epochs, "learning_rate": lr, "batch_size": batch_size},
    # )

    optimizer = torch.optim.AdamW(params=train_model.parameters(), lr=lr)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=0.6,
        max_tokens=max_completion_length,
        prompt_logprobs=1,
        logprobs=1,
        include_stop_str_in_output=True,
        stop_token_ids=[eos_token_id],
    )

    loss_fn: LossMethod = grpo

    pbar = tqdm(total=epochs)
    total_steps = len(train_loader)

    # clip all inputs prompts to 512 length

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            pbar.set_description(f"Epoch: {epoch + 1}: {step + 1}/{total_steps}")

            # generate samples for all prompts in this batch
            print(sampling_params)
            groups = rollout(batch["prompt"], sampling_params, debug=True)

            to_log = {}
            to_log["prompt_lengths"] = [
                len(x) for x in tokenizer(batch["prompt"])["input_ids"]
            ]
            to_log["completion_lengths"] = [
                [(len(s["completion_tokens"])) for s in g["samples"]] for g in groups
            ]

            total_reward = 0.0
            step_loss = 0.0

            # find the rewards and losses for each group
            for g_idx, g in enumerate(groups):
                if g["max_completion_length"] < 100:
                    print(
                        f"Skipping group with low completion length: {g['max_completion_length']}"
                    )
                    print([s["completion_string"] for s in g["samples"]])
                    continue

                total_group_reward, group_loss = process_group(g, g_idx, loss_fn)
                step_loss += group_loss
                total_reward += total_group_reward

                gpu_mem_allocated("after samples loop and clearing")
                # Force cleanup after each group
                torch.cuda.empty_cache()

            # Gradient update + vllm update
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                print(f"Optimizer size: {optimizer_state_size_mb(optimizer)}")

            if (step + 1) % (100) == 0:
                print("Updating vLLM weights")
                start = time.time()
                vllm.collective_rpc(update_vllm_weights)
                end = time.time()
                vllm.reset_prefix_cache()
                print(f"Time taken to update vLLM weights: {end - start} seconds")

            avg_reward = total_reward / (batch_size * n_samples)

            # wandb.log(
            #     {
            #         "epoch": epoch,
            #         "avg_reward": avg_reward,
            #         "loss": step_loss / batch_size,
            #         **to_log,
            #     },
            #     step + 1,
            # )


train()
