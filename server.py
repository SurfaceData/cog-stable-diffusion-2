"""Configures a simple FastAPI server around Stable Diffusion V2"

This is partially based on the Stable Diffusion V2 Space on HuggingFace [1].
It uses recent updates to diffusers to run the model.  Additionally it supports
cross origin requests while also requiring calls to have an API key.

[1]: https://huggingface.co/spaces/stabilityai/stable-diffusion
"""
import cog
import io
import tempfile
import torch
import typing

from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
from secrets import token_hex

REPO_ID = "stabilityai/stable-diffusion-2"
device = "cuda"

class Predictor(cog.BasePredictor):
    @torch.inference_mode()
    def setup(self):
        """Download and load the model with memory efficient attention."""

        # Create a new API key and log it to the console.
        self.apikey = token_hex(16)
        print(f'Server API Key: {self.apikey}')

        # Load the things.
        scheduler = EulerDiscreteScheduler.from_pretrained(
                REPO_ID, subfolder="scheduler", prediction_type="v_prediction")
        pipe = DiffusionPipeline.from_pretrained(
                REPO_ID, torch_dtype=torch.float16, revision="fp16", scheduler=scheduler)
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
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
        apikey: str = cog.Input(description="API Access Key.", default=""),
        prompt: str = cog.Input(description="Your text prompt.", default=""),
        guidance_scale: float = cog.Input(
            default=5.0,
            description="Classifier-free guidance scale. Higher values will result in more guidance toward caption, with diminishing returns. Try values between 1.0 and 40.0. In general, going above 5.0 will introduce some artifacting.",
            le=100.0,
            ge=-20.0,
        ),
        steps: int = cog.Input(
            default=50,
            description="Number of diffusion steps to run. Due to PLMS sampling, using more than 100 steps is unnecessary and may simply produce the exact same output.",
            le=250,
            ge=15,
        ),
        batch_size: int = cog.Input(
            default=4,
            description="Batch size. (higher = slower)",
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
    ) -> typing.Iterator[cog.Path]:
        """Generate samples from Stable Diffusion and return them."""

        # Reject any request with an invalid API key.
        if apikey != self.apikey:
            return []

        # Seed the generator and predict images.
        generator = torch.Generator(device=device).manual_seed(seed)
        images = self.pipe(prompt,
                      width=width,
                      height=height,
                      num_inference_steps=steps,
                      guidance_scale=guidance_scale,
                      num_images_per_prompt=batch_size,
                      generator=generator).images

        # Emit the generated images.  This is modeled after how ldm-finetune
        # returns their results.
        results = []
        temp_path = cog.Path(tempfile.mkdtemp())
        for idx, image in enumerate(images):
            output_path = temp_path / f"{idx}.png"
            image.save(output_path, format='png')
            results.append(cog.Path(output_path))
        return results
