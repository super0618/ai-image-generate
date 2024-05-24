import diffusers
import transformers

import sys
import os
import shutil
import time

import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

if torch.cuda.is_available():
    device_name = torch.device("cuda")
    torch_dtype = torch.float16
else:
    device_name = torch.device("cpu")
    torch_dtype = torch.float32

# Follows community convention.
# Clip skip = 1 uses the all text encoder layers.
# Clip skip = 2 skips the last text encoder layer.

clip_skip = 1

if clip_skip > 1:
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder = "text_encoder",
        num_hidden_layers = 12 - (clip_skip - 1),
        torch_dtype = torch_dtype
    )

# Load the pipeline.

model_path = "ChineseLandscapeArt_v10"

if clip_skip > 1:
    # TODO clean this up with the condition below.
    pipe = diffusers.DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype = torch_dtype,
        safety_checker = None,
        text_encoder = text_encoder,
    )
else:
    pipe = diffusers.DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype = torch_dtype,
        safety_checker = None
    )

pipe = pipe.to(device_name)

# Change the pipe scheduler to EADS.
pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config
)

# Prompt embeddings to overcome CLIP 77 token limit.
# https://github.com/huggingface/diffusers/issues/2136

def get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character = ",",
    device = torch.device("cpu")
):
    max_length = pipe.tokenizer.model_max_length
    # Simple method of checking if the prompt is longer than the negative
    # prompt - split the input strings using `split_character`.
    count_prompt = len(prompt.split(split_character))
    count_negative_prompt = len(negative_prompt.split(split_character))

    # If prompt is longer than negative prompt.
    if count_prompt >= count_negative_prompt:
        input_ids = pipe.tokenizer(
            prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipe.tokenizer(
            negative_prompt,
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length,
            return_tensors = "pt"
        ).input_ids.to(device)

    # If negative prompt is longer than prompt.
    else:
        negative_ids = pipe.tokenizer(
            negative_prompt, return_tensors = "pt", truncation = False
        ).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipe.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation = False,
            padding = "max_length",
            max_length = shape_max_length
        ).input_ids.to(device)

    # Concatenate the individual prompt embeddings.
    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(
            pipe.text_encoder(input_ids[:, i: i + max_length])[0]
        )
        neg_embeds.append(
            pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
        )

    return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)

prompt = """beautiful Chinese Landscape Art, best quality, intricate,
water colors, snowy mountains, glacier, snow, starry night sky, stars,
milkyway"""


negative_prompt = """deformed, weird, bad resolution, bad depiction,
not Chinese style, weird, has people, worst quality, worst resolution,
too blurry, not relevant"""

prompt_embeds, negative_prompt_embeds = get_prompt_embeddings(
    pipe,
    prompt,
    negative_prompt,
    split_character = ",",
    device = device_name
)


# Set to True to use prompt embeddings, and False to
# use the prompt strings.
use_prompt_embeddings = True

# Seed and batch size.
start_idx = 0
batch_size = 1
seeds = [i for i in range(start_idx , start_idx + batch_size, 1)]

# Number of inference steps.
num_inference_steps = 20

# Guidance scale.
guidance_scale = 7

# Image dimensions - limited to GPU memory.
width  = 512
height = 512

images = []

for count, seed in enumerate(seeds):
    start_time = time.time()

    if use_prompt_embeddings is False:
        new_img = pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            width = width,
            height = height,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            num_images_per_prompt = 1,
            generator = torch.manual_seed(seed),
        ).images
    else:
        new_img = pipe(
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            width = width,
            height = height,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            num_images_per_prompt = 1,
            generator = torch.manual_seed(seed),
        ).images


    images = images + new_img

# Plot pipeline outputs.
def plot_images(images, labels = None):
    N = len(images)
    n_cols = 2
    n_rows = int(np.ceil(N / n_cols))

    plt.figure(figsize = (20, 5 * n_rows))
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(np.array(images[i]))
        plt.axis(False)
    plt.show()

plot_images(images, seeds[:len(images)])