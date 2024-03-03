import gradio as gr
from PIL import Image
import numpy as np
import torch
from transformers import pipeline
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image

# Pre-load required models
depth_estimator = pipeline('depth-estimation')
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)
sd_pipeline = StableDiffusionControlNetPipeline.from_single_file(
    "/content/CyberRealistic_V4.1_FP16.safetensors", controlnet=controlnet, torch_dtype=torch.float16
)
sd_pipeline.to("cuda")  # Assuming you have a CUDA-enabled GPU

# Function to process depth map
def get_depth_map(depth_map_image):
    """Processes a depth map image for use with ControlNet.

    Args:
        depth_map_image (PIL.Image): The input depth map image.

    Returns:
        PIL.Image: The processed depth map image, ready for ControlNet.
    """
    depth_map = depth_estimator(depth_map_image)['depth']
    depth_map = np.array(depth_map)
    depth_map = depth_map[:, :, None]
    depth_map = np.concatenate([depth_map, depth_map, depth_map], axis=2)
    depth_map = depth_map.astype(np.uint8)  # Convert to uint8 for image compatibility
    return Image.fromarray(depth_map)

# Function to process IP adapter image
def get_ip_adapter_image(ip_adapter_image):
    """Loads the IP adapter for use with the image generation pipeline.

    Args:
        ip_adapter_image (PIL.Image): The IP adapter image.

    Returns:
        PIL.Image: The IP adapter image (returned directly).
    """
    sd_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
    sd_pipeline.set_ip_adapter_scale(0.7)
    return ip_adapter_image

# Function to resize images
def resize_image(image):
    """Resizes an image while ensuring the shortest side is 512 pixels.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The resized image.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    width, height = image.size
    max_size = max(width, height)
    scale_factor = 768 / max_size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return image.resize((new_width, new_height))

# Function to generate the image
def generate_image(
    prompt,
    negative_prompt,
    #width,
    #height,
    depth_map_image,
    ip_adapter_image,
    num_inference_steps=30,
    seed=33,
):
    depth_map_image = resize_image(depth_map_image)
    ip_adapter_image = resize_image(ip_adapter_image)

    depth_map = get_depth_map(depth_map_image)
    ip_adapter_image = get_ip_adapter_image(ip_adapter_image)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = sd_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        #width=width,
        #height=height,
        image=depth_map,
        ip_adapter_image=ip_adapter_image,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    return images

# Create the Gradio interface
with gr.Blocks(title="Interior Renovation Ai") as demo:
    gr.Markdown("""# Interior Renovation Ai""")   
    gr.Markdown("โปรแกรมสำหรับรีโนเวทห้องด้วย Ai | fb.com/PromptAlchemist")  
    with gr.Row():
        prompt = gr.Textbox(lines=1, placeholder="เขียนพรอมต์ที่ต้องการเช่น Living room", label="Prompt")
        negative_prompt = gr.Textbox(lines=1, value="Low quality, bad quality, worst quality, 3d, cartoon, painting", label="Negative Prompt")

    with gr.Row():
        depth_map = gr.Image(label="Room", type="pil", height=500, width=500)
        ip_adapter_image = gr.Image(label="Reference", type="pil", height=500, width=500)
        output_image = gr.Image(label="Output", type="pil", height=500, width=500) 

    steps = gr.Slider(0, 100, value=30, label="Steps")
    seed = gr.Number(value=33, label="Seed")
    generate_button = gr.Button("Generate Image")

    generate_button.click(fn=generate_image, 
                          inputs=[prompt, negative_prompt, depth_map, ip_adapter_image, steps, seed],
                          outputs=output_image) 

demo.launch(share=True)
