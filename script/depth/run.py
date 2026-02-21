# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import numpy as np
import os
import torch
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from marigold.ramit_model.ramit import RAMiTCond
from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
from marigold import iGlemDepthPipeline
from src.util.seeding import seed_all
from safetensors.torch import load_file as safe_load_file

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def get_args():
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold: Monocular Depth Estimation Multi-image Inference"
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default="prs-eth/marigold-depth-v1-1",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--finetune_checkpoint",
        type=str,
        default="output/train_weather_depth/checkpoint/latest/unet/diffusion_pytorch_model.safetensors",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="restore",
        help="[concat|original|restore]",
    )
    parser.add_argument(
        "--ramit_checkpoint",
        type=str,
        default="output/train_rasmit_latent/checkpoint/iter_080000/ramit.pth",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        default="input",
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output/example", 
        help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=5,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. If set to "
        "`None`, default value will be read from checkpoint.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Resolution to which the input is resized before performing estimation. `0` uses the original input "
        "resolution; `None` resolves the best default from the model checkpoint. Default: `None`",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled. Default: `1`.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Setting this flag will output the result at the effective value of `processing_res`, otherwise the "
        "output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or "
        "`nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to visualize depth predictions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Use Apple Silicon for faster inference (subject to availability).",
    )

    args = parser.parse_args()
    return args


def get_pipeline(args):
    """
        Get the pipeline for specific models
    """
    if args.version == 'concat':
        unet = UNet2DConditionModel.from_config(args.base_checkpoint, subfolder="unet")
        unet.conv_in = RAMiTCond()
        
        # Load the trained checkpoint weights
        if args.finetune_checkpoint.endswith(".safetensors"):
            state_dict = safe_load_file(args.finetune_checkpoint)
        else:
            state_dict = torch.load(args.finetune_checkpoint, map_location='cpu')
            
        unet.load_state_dict(state_dict)
        
        vae = AutoencoderKL.from_pretrained(args.base_checkpoint, subfolder="vae")
        scheduler = DDIMScheduler.from_pretrained(args.base_checkpoint, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(args.base_checkpoint, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(args.base_checkpoint, subfolder="tokenizer")
        
        pipeline = iGlemDepthPipeline(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            variant=args.variant, 
            torch_dtype=args.dtype,
        )
        
    elif args.version == 'adapter':
        vae = AutoencoderKL.from_pretrained(args.base_checkpoint, subfolder="vae")
        scheduler = DDIMScheduler.from_pretrained(args.base_checkpoint, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(args.base_checkpoint, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(args.base_checkpoint, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(args.finetune_checkpoint, subfolder="unet")
        adapter = RAMiTCond.from_pretrained(args.finetune_checkpoint, subfolder="adapter")
        pipeline = iGlemDepthPipeline(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            adapter=adapter,
        )
    
    elif args.version == 'original':
        # Use Original MarigoldDepthPipeline
        pipeline: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
            args.base_checkpoint, 
            variant=args.variant, 
            torch_dtype=args.dtype
        )
        
    else:
        raise NotImplementedError
    
    return pipeline
      
    
if "__main__" == __name__:
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, "
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        args.dtype = torch.float16
        args.variant = "fp16"
        logging.warning(
            f"Running with half precision ({args.dtype}), might lead to suboptimal result."
        )
    else:
        args.dtype = torch.float32
        args.variant = None

    pipeline = get_pipeline(args)

    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipeline = pipeline.to(device)
    logging.info(
        f"Loaded depth pipeline: scale_invariant={pipeline.scale_invariant}, shift_invariant={pipeline.shift_invariant}"
    )
    # Move RAMiT module to the same device if it exists
    if hasattr(pipeline, 'ramit_module') and pipeline.ramit_module is not None:
        pipeline.ramit_module = pipeline.ramit_module.to(device)
    
    logging.info(
        f"Loaded depth pipeline: scale_invariant={pipeline.scale_invariant}, shift_invariant={pipeline.shift_invariant}"
    )
    
    # Print out config
    logging.info(
        f"with denoise_steps = {denoise_steps or pipeline.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipeline.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(rgb_filename_list, desc="Depth Inference", leave=True):
            # Read input image
            input_image = Image.open(rgb_path)

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Perform inference
            pipe_out: MarigoldDepthOutput = pipeline(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_depth"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)

            # Save as 16-bit uint png
            depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            # Colorize
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )
            depth_colored.save(colored_save_path)
