import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionLoRAForDreamBoothPipeline
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image
import os

# Setup
model_id = "runwayml/stable-diffusion-v1-5"
output_dir = "./furniture_dreambooth_lora"
train_dir = "./resized"  # <-- your resized images folder

# Load pretrained model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda")

# Prepare dataset
train_data = []
for img_file in os.listdir(train_dir):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        caption_file = os.path.splitext(img_file)[0] + ".txt"
        caption_path = os.path.join(train_dir, caption_file)
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                caption = f.read().strip()
            train_data.append((os.path.join(train_dir, img_file), caption))

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["transformer_blocks.0.attn1.to_q", "transformer_blocks.0.attn1.to_v"],
    lora_dropout=0.1,
    task_type="TEXT_TO_IMAGE"
)

pipe.unet = get_peft_model(pipe.unet, lora_config)

# Dummy training loop (small-scale)
# In reality, you'd use Trainer / Accelerate for full training
from torch.optim import AdamW
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

optimizer = AdamW(pipe.unet.parameters(), lr=1e-5)

for epoch in range(20):  # you can increase this number
    total_loss = 0
    for img_path, caption in train_data:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to("cuda", dtype=torch.float16)

        # Encode caption
        tokens = pipe.tokenizer(
            caption,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to("cuda")

        # Encode latents
        latents = pipe.vae.encode(image_tensor).latent_dist.sample() * 0.18215

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (1,), device="cuda").long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Forward pass
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=pipe.text_encoder(tokens)).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")

# Save final model
pipe.save_pretrained(output_dir)
print("🚀 Training complete. Model saved.")
