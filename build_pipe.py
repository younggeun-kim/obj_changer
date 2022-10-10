import inspect
import warnings
from typing import List, Optional, Union
from io import BytesIO
import requests
import numpy as np
import PIL
from PIL import Image

import torch
from torch import autocast
from tqdm.auto import tqdm

from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionImg2ImgPipeline

def make_models():
  device = "cuda"
  model_id_or_path = "CompVis/stable-diffusion-v1-4"
  pipe1 = StableDiffusionInpaintPipeline.from_pretrained(
      model_id_or_path,
      revision="fp16", 
  )
  # or download via git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
  # and pass `model_id_or_path="./stable-diffusion-v1-4"`.
  pipe1 = pipe1.to(device)
  pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(
      model_id_or_path,
      revision="fp16", 
      torch_dtype=torch.float16,
      use_auth_token=True
  )
  pipe2 = pipe2.to(device)
  return [pipe1, pipe2]

if __name__ == __main__:
  pipe = make_models()