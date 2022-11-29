"""Configures a simple FastAPI server around Stable Diffusion V2"

This is partially based on the Stable Diffusion V2 Space on HuggingFace [1].
It uses recent updates to diffusers to run the model.  Additionally it supports
cross origin requests while also requiring calls to have an API key.

[1]: https://huggingface.co/spaces/stabilityai/stable-diffusion
"""
import cog
import cv2
import io
import numpy as np
import tempfile
import torch
import typing

from imwatermark import WatermarkEncoder
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, PNDMScheduler, LMSDiscreteScheduler
from random import randint
from secrets import token_hex
from PIL import Image

REPO_ID = "stabilityai/stable-diffusion-2-base"
WATERMARK = "SDV2"
device = "cuda"

class Predictor(cog.BasePredictor):
    @torch.inference_mode()
    def setup(self):
        """Download and load the model with memory efficient attention."""

        # Create a new API key and log it to the console.
        self.apikey = token_hex(16)
        print(f'Server API Key: {self.apikey}')

        # Create the watermark encoder
        self.wm_encoder = WatermarkEncoder()
        self.wm_encoder.set_watermark('bytes', WATERMARK.encode('utf-8'))

        # Load the things.
        pipe = DiffusionPipeline.from_pretrained(
                REPO_ID, torch_dtype=torch.float16, revision="fp16")
        pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config)
        pipe = pipe.to(device)
        self.pipe = pipe


    def configure_api(self, app):
        """Allow for cross origin requests."""
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @torch.inference_mode()
    def predict(
        self,
        apikey: str = cog.Input(
            default="",
            description="API Access Key."), 
        prompt: str = cog.Input(
            default="",
            description="Your text prompt."),
        negative_prompt: str = cog.Input(
            default="",
            description="The prompts guiding what not to generate."),
        sampler: str = cog.Input(
            default="dpm",
            description="The sampling scheduler to use to generate images.",
            choices=["dpm", "ddim", "euler", "euler_a", "pndm", "lms"],
            ),
        guidance_scale: float = cog.Input(
            default=5.0,
            description="Classifier-free guidance scale. Higher values will result in more guidance toward caption, with diminishing returns. Try values between 1.0 and 40.0. In general, going above 5.0 will introduce some artifacting.",
            le=100.0,
            ge=-20.0,
        ),
        steps: int = cog.Input(
            default=50,
            description="Number of diffusion steps to run.",
            le=250,
            ge=15,
        ),
        batch_size: int = cog.Input(
            default=4,
            description="The number of images to generate.",
            ge=1,
            le=16,
        ),
        width: int = cog.Input(
            default=512,
            description="Target width",
        ),
        height: int = cog.Input(
            default=512,
            description="Target height",
        ),
        seed: int = cog.Input(
            default=-1,
            description="Seed for random number generator. If -1, a random seed will be chosen.",
            ge=-1,
            le=(2**32 - 1),
        ),
        eta: float = cog.Input(
            default=0.0,
            description="Meta parameter to the DDIM scheduler",
        ),
    ) -> typing.Iterator[cog.Path]:
        """Generate samples from Stable Diffusion and return them."""

        # Reject any request with an invalid API key.
        if apikey != self.apikey:
            return []

        # Predict images.
        seed = seed if seed != -1 else randint(0, 2**32 -1)
        generator = torch.Generator(device=device).manual_seed(seed)

        if sampler == 'euler':
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipe.scheduler.config)
        elif sampler == 'euler_a':
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    self.pipe.scheduler.config)
        elif sampler == 'dpm':
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config)
        elif sampler == 'pndm':
            self.pipe.scheduler = PNDMScheduler.from_config(
                    self.pipe.scheduler.config)
        elif sampler == 'lms':
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(
                    self.pipe.scheduler.config)
        elif sampler == 'ddim':
            self.pipe.scheduler = DDIMScheduler.from_config(
                    self.pipe.scheduler.config)

        images = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=batch_size).images

        # Emit the generated images.  This is modeled after how ldm-finetune
        # returns their results.
        results = []
        temp_path = cog.Path(tempfile.mkdtemp())
        for idx, image in enumerate(images):
            image = self.put_watermark(image)
            output_path = temp_path / f"{idx}.png"
            image.save(output_path, format='png')
            results.append(cog.Path(output_path))
        return results

    def put_watermark(self, image):
        if self.wm_encoder is None:
            return image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = self.wm_encoder.encode(image, 'dwtDct')
        return Image.fromarray(image[:,:,::-1])
