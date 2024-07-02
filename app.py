import streamlit as st
import torch
import torchvision.utils as vutils
import numpy as np
from torch import nn
from PIL import Image

# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Load the pre-trained generator model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator (1).h5', map_location=device))
generator.eval()

# Define the Streamlit app
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Pokemon Image Generator")

# Generate and display images
num_images = st.slider("Number of images to generate", 1, 64, 16, 1)
noise = torch.randn(num_images, 100, 1, 1, device=device)
generated_images = generator(noise)

# Convert tensor to numpy array and resize images
generated_images = (generated_images.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2
generated_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in generated_images]

# Display the generated images as an animated GIF
st.image(generated_images, use_column_width=True, caption="Generated Pokemon Images")
