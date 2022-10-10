import inspect
import warnings
import argparse
from typing import List, Optional, Union
from io import BytesIO
import requests
import numpy as np
import PIL
from PIL import Image
import time

import torch
from torch import autocast
from tqdm.auto import tqdm

from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionImg2ImgPipeline
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def make_mask(boxes):
  y1, x1, y2, x2 = boxes
  mask_img = np.zeros ((512, 512, 3)).astype(np.uint8)
  mask_img[y1:y2, x1:x2, :] = 255
  mask_img = Image.fromarray(mask_img)
  return mask_img

def object_changer(init_image, boxes, prompt, pipe):
  pipe1, pipe2 = pipe
  generator = torch.Generator(device=device).manual_seed(1024)
  mask_img = make_mask(boxes)
  image_processed = pipe1(prompt=prompt, init_image=init_image, mask_image=mask_img, strength=0.65).images
  with autocast("cuda"):
      images = pipe2(prompt=prompt, init_image=image_processed[0], strength=0.4, guidance_scale=7.5, generator=generator).images[0]
  return image_processed[0], images

def make_models(device=device):
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

pipe = make_models(device)

def inference():
    
    url = "https://images.unsplash.com/photo-1587502536575-6dfba0a6e017?ixlib=rb-1.2.1&ixid=MnwxMjA3fDF8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2864&q=80"
    prompt = "a person holding a red coke near the beach" 
    boxes = [160, 200, 370, 320]
    init_image = download_image(url).resize((512, 512))
    total_start = time.time()
    image_processed, images = object_changer(init_image, boxes, prompt, pipe)
    total_time = time.time() - total_start

    return {
        'result': images,
        'total_time': total_time,
    }




"""
if __name__ == __main__:
    parser = argparse.ArgumentParser(
        description='brain image classification client')
    parser.add_argument('-r', '--url',
                        nargs='?',
                        type=str,
                        required=True)
    parser.add_argument('-i', '--boxes',
                        type=list,
                        required=True)
    parser.add_argument('-i', '--prompt',
                        type=str,
                        required=True)
    parser.add_argument('-i', '--pipe',
                        type=list,
                        required=True)                       
    args = parser.parse_args()
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    url = "https://images.unsplash.com/photo-1587502536575-6dfba0a6e017?ixlib=rb-1.2.1&ixid=MnwxMjA3fDF8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2864&q=80"

    prompt = "a person holding a red coke near the beach" 
    boxes = [160, 200, 370, 320]

    init_image = download_image(url).resize((512, 512))
    image_processed, images = object_changer(init_image, boxes, prompt, pipe)
    images.save("img.png")
    """