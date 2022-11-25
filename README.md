# COG Stable Diffusion V2

## Introduction

This [Cog](https://github.com/replicate/cog) server runs Stability AI's
[V2](https://stability.ai/blog/stable-diffusion-v2-release) release.

This allows for cross origin requests and requires API calls to have a matching
API key that's generated on setup.

## Usage

Generate the image with:

```sh
cog build -t surface-data/stable-diffusion-v2
```

Then deploy with docker as normal:

```sh
docker run -d -p 5000:5000 --gpus all surface-data/stable-diffusion-v2
```
