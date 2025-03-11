import torch
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model and move it to GPU
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).to(device)

# Define the diffusion model and move it to GPU
diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,  # number of steps
).to(device)

# Generate training images and move them to GPU
training_images = torch.randn(8, 3, 128, 128, device=device)

# Compute loss and perform backpropagation
loss = diffusion(training_images)
loss.backward()

# Sample images (ensure tensors are on GPU)
sampled_images = diffusion.sample(batch_size=4).cpu().permute(0, 2, 3, 1) 
print("Sampled images shape:", sampled_images.shape)

def show_images(images):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i].numpy().clip(0, 1))  # Normalize values between 0 and 1
        plt.axis("off")
    plt.show()

# Display sampled images
show_images(sampled_images)
