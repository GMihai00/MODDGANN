from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.cuda.empty_cache()

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)


diffusion = GaussianDiffusion(
    model,
    image_size = 160,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/home/gmihai00/Repos/TenserflowModelTraining/bucal_cavity_diseases_dataset/google',
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 1000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    # save_and_sample_every = 1000,     # how often to save and sample from the model
)

trainer.train()

trainer.save(milestone=2)

# trainer.load(milestone=1)

 # MANUAL CHANGE!
        # if torch.cuda.is_available():
        #     try:
        #             out = F.scaled_dot_product_attention(
        #                 q, k, v, dropout_p = self.dropout if self.training else 0.
        #             )
        #     except Exception as e:
        #         print("Failed to use scaled_dot_product_attention:", e)
        #         # Fallback to normal attention if the optimized kernel isn't available
        #         out = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        #         out = F.softmax(out, dim=-1)
        #         out = torch.matmul(out, v)
        # else:
        #     # For CPU, we revert to the regular attention mechanism
        #     out = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        #     out = F.softmax(out, dim=-1)
        #     out = torch.matmul(out, v)
        
# INSTEAD OF
        # with torch.backends.cuda.sdp_kernel(**config._asdict()):
        #     out = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p = self.dropout if self.training else 0.
        #     )


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
