#Don't forget to source venv/bin/activate your virtual environment
#And to login to your hugginface
#pip install huggingface_hub
#huggingface-cli login --token [YOURTOKEN]
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipe.to("mps")
pipe.enable_attention_slicing()
pipe.safety_checker = None
pipe.requires_safety_checker = False

style= " masterpiece, intricate, elegant futuristic wardrobe, highly detailed, digital painting, artstation, concept art, crepuscular rays, smooth, sharp focus, illustration,"
prompt = f"Astronaut on a savage planet. {style}"
negative= "duplicate, blurry, deformed, text, missing limbs, extra limbs, malformed limbs, mutilated, out of frame"

print(f"Executing on prompt {prompt}")

generator = torch.Generator(device="mps")
for seed in range(1,32,1):
  
  generator = generator.manual_seed(seed)
  images = pipe(prompt=prompt, negative_prompt=negative,num_inference_steps=20,generator = generator).images

  for image in images:  
    image.save(f"astronaut_in_a_junglex-seed-{seed}.png")

