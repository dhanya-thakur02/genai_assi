from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

def build_prompt(subject, topic):
    prompt = f"A children's educational book illustration representing {topic}, a {subject} concept, simple cartoon style, clear and easy to understand, single centered subject, vibrant colors, white background"
    negative_prompt = "text, words, letters, writing, labels, captions, title, watermark, signature, diagram, chart, infographic, poster, pattern, repeating, tiled, seamless, wallpaper, abstract art, blurry, low quality"
    return prompt, negative_prompt

prompt, negative_prompt = build_prompt("Biology", "photosynthesis")
generator = torch.Generator("cuda").manual_seed(17)
image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=9, generator=generator).images[0]
image.save("final_test_1.png")
print("Saved final_test_1.png (Biology/photosynthesis)")

print("Done!")