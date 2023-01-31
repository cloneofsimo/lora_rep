import os
from typing import List
import time

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from lora_diffusion import patch_pipe, tune_lora_scale
from safetensors.torch import safe_open, save_file

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


def download_lora(url):
    from hashlib import sha512

    fn = sha512(url.encode()).hexdigest() + ".safetensors"

    if not os.path.exists(fn):
        import requests

        print("Downloading LoRA model... from", url)
        # stream chunks of the file to disk
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    else:
        print("Using disk cache...")

    return fn


def lora_add(path_1, alpha_1, path_2, alpha_2, output_path="output.safetensors"):
    """Scales each lora by appropriate weights & returns"""
    safeloras_1 = safe_open(path_1, framework="pt", device="cpu")
    safeloras_2 = safe_open(path_2, framework="pt", device="cpu")

    metadata = dict(safeloras_1.metadata())
    metadata.update(dict(safeloras_2.metadata()))

    ret_tensor = {}

    for keys in set(list(safeloras_1.keys()) + list(safeloras_2.keys())):
        if keys.startswith("text_encoder") or keys.startswith("unet"):

            tens1 = safeloras_1.get_tensor(keys)
            tens2 = safeloras_2.get_tensor(keys)

            tens = alpha_1 * tens1 + alpha_2 * tens2
            ret_tensor[keys] = tens
        else:
            if keys in safeloras_1.keys():

                tens1 = safeloras_1.get_tensor(keys)
            else:
                tens1 = safeloras_2.get_tensor(keys)

            ret_tensor[keys] = tens1

    # we don't need to go to-> from safetensors here, adding in now for compat's sake
    start_time = time.time()
    save_file(ret_tensor, output_path, metadata)
    print(f"saving time: {time.time() - start_time}")
    return output_path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        self.loaded = None

    def merge_loras(self, url_1, scale_1, url_2, scale_2):
        merged_lora_ref = f"{a}-{b}-{scale_1}-{scale-2}"
        if self.loaded == merged_lora_ref:
            print("The requested two LoRAs are already scaled and loaded.")
            return

        lora_1 = download_lora(url_1)
        lora_2 = download_lora(url_2)

        st = time.time()
        local_lora_path = lora_add(lora_1, scale_1, lora_2, scale_2)
        print(f"merging time: {time.time() - st}")

        patch_pipe(self.pipe, local_lora_path)
        # merging tunes lora scale so we don't need to do that here.
        self.loaded = merged_lora_ref

    def load_lora(self, url, scale):
        if url == self.loaded:
            print("The requested LoRA model is already loaded...")
            return

        start_time = time.time()
        local_lora_safetensors = download_lora(url)
        print("download_lora time:", time.time() - start_time)

        start_time = time.time()
        patch_pipe(self.pipe, local_lora_safetensors)
        print("patch_pipe time:", time.time() - start_time)

        start_time = time.time()
        tune_lora_scale(self.pipe.unet, scale)
        tune_lora_scale(self.pipe.text_encoder, scale)
        print("tune_lora_scale time:", time.time() - start_time)

        self.loaded = url

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        lora_1_url: str = Input(
            description="url for safetensors of lora model.",
        ),
        lora_1_scale: float = Input(
            description="LoRA scale for weight interpolation",
            ge=0.0,
            le=4.0,
            default=0.8,
        ),
        lora_2_url: str = Input(
            description="(Optional) url for safetensors of second lora model.",
        ),
        lora_2_scale: float = Input(
            description="(Optional) LoRA scale for weight interpolation, lora_1_scale*lora_1 + lora_2_scale*lora_2. Scales don't have to sum to 1.",
            ge=0.0,
            le=4.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        if not lora_1_url:
            raise ValueError("Please specify a LoRA model url.")

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        if lora_2_url:
            self.merge_loras(lora_1_url, lora_1_scale, lora_2_url, lora_2_scale)
        else:
            self.load_lora(lora_1_url, lora_1_scale)

        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
