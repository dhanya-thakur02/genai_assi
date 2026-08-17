import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import ollama

print("Loading Stable Diffusion pipeline")
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")
print("Pipeline loaded.")

def build_prompt(subject, topic):
    prompt = f"A children's educational book illustration representing {topic}, a {subject} concept, simple cartoon style, clear and easy to understand, single centered subject, vibrant colors, white background"
    negative_prompt = "text, words, letters, writing, labels, captions, title, watermark, signature, diagram, chart, infographic, poster, pattern, repeating, tiled, seamless, wallpaper, abstract art, blurry, low quality"
    return prompt, negative_prompt

def explain_image(image_path, subject, topic):
    response = ollama.chat(
        model='qwen2.5vl:3b',
        messages=[{
            'role': 'user',
            'content': f"This illustration was generated to represent the topic '{topic}' in {subject}. Look at the image and explain, in 2-3 simple sentences suitable for a student, how it relates to {topic}.",
            'images': [image_path]
        }]
    )
    return response['message']['content']

def generate(subject, topic):
    prompt, negative_prompt = build_prompt(subject, topic)
    generator = torch.Generator("cuda").manual_seed(17)
    image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=9, generator=generator).images[0]
    image.save("generated_image.png")
    explanation = explain_image("generated_image.png", subject, topic)
    return image, explanation

with gr.Blocks(title="Educational Image Generator") as demo:
    gr.Markdown("# LLM-Assisted Educational Image Generation")
    with gr.Row():
        subject = gr.Dropdown(
            ["Biology", "Physics", "Chemistry", "Geography", "General Science"],
            label="Subject"
        )
        topic = gr.Textbox(label="Topic", placeholder="e.g. Photosynthesis")
    generate_btn = gr.Button("Generate", variant="primary")
    with gr.Row():
        image_output = gr.Image(label="Generated Illustration")
        explanation_output = gr.Textbox(label="Explanation", lines=6)

    generate_btn.click(fn=generate, inputs=[subject, topic], outputs=[image_output, explanation_output])

demo.launch()