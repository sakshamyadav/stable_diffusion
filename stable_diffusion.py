# Install diffusers library in colab
!pip install -q diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git

# Install pytorch in colab
!pip install torch torchvision

# Import necessary libraries and modules
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torch 
import random
import time
from google.colab import files

# Upload text file containing model names to colab
models = files.upload()

# Upload the 17 images to colab
images = files.upload()

Get model file from colab
model_file = list(models.keys())[0]

# Open the model file in read mode
with open(model_file, "r") as f:
    # read all lines and store them in a list
    lines = f.readlines()

# Remove newline characters from each line and store the model names in a list
model_list = [line.strip() for line in lines]

# Check if CUDA is available, and if so, use it, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open a file called "metrics.txt" in write mode to record metrics
metrics_file = open("metrics.txt", "w")

# Loop over each image in the "images" dictionary
for image in images.keys():

    # Load the input image and convert it to RGB format
    init_image = Image.open(BytesIO(images[image])).convert("RGB")
    
    # Resize the image to 768x512 pixels
    init_image = init_image.resize((768, 512))
    
    # Select a random model ID from the "model_list"
    model_id = random.choice(model_list)

    # Initialize a StableDiffusionImg2ImgPipeline object with the selected model ID
    # and set the torch data type to float16
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # Move the pipeline object to the selected device (CPU or GPU)
    pipe = pipe.to(device)
    
    # Set the prompt text for the image generation
    prompt = "A fantasy landscape, trending on artstation"

    # Record the start time
    start_time = time.time()

    # Generate the output image using the pipeline object with the given prompt,
    # input image, strength, and guidance scale
    output_images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time in seconds
    elapsed_time = end_time - start_time
    
    # Get the GPU memory usage in megabytes
    memory_usage = torch.cuda.memory_allocated() / 1e6
        
    # Write the metrics for this image to the metrics file
    metrics_file.write(f"Image {image}: Time = {elapsed_time:.2f} s, Memory = {memory_usage:.2f} MB\n")
    
    # Save the output image to a file with a name based on the input image name
    output_images[0].save(f"output_{image}.png")

# Close the metrics file
metrics_file.close()
