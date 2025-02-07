#%%
import torch as th
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

#%%

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
])

train_dataset  = datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform)
trainloader = th.utils.data.DataLoader(train_dataset, batch_size=100,shuffle=True)

#%% Show sample image

# Function to show images
def show_image(image, label=None):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

#%% Show images
index = 0
image = images[index].numpy().squeeze()
show_image(image)
# %% Mean of image points

plt.figure(figsize=(8, 6))
plt.hist(image.flatten(), bins=40)
plt.show()

# %% Adding noise

image = images[index].squeeze(0)
steps = 1

for i in range(steps):
    noise = th.randn_like(image)
    image = image + noise

plt.figure(figsize=(8, 6))
plt.hist(image.flatten(), bins=40)
plt.show()

show_image(image)

# %% 

def linear_beta_schedule(diffusion_timesteps):
    scale = 1000 / diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return th.linspace(beta_start, beta_end, diffusion_timesteps)

steps = 300
betas = linear_beta_schedule(steps)
plt.plot(betas)
plt.show()

# %% One step noise adding


def p_sample(steps, img, t):
    B = img.size(0)
    noise = th.randn_like(img) #ground truth noise

    betas = linear_beta_schedule(steps).float()
    alphas = 1. - betas
    alpha_prod = th.cumprod(alphas, 0)
    alpha_prod_prev = th.cat([th.tensor([1.0]), alpha_prod[:-1]])

    a = th.sqrt(alpha_prod)[t]
    b = th.sqrt(1- alpha_prod)[t]
    x_t = a*img + b*noise

    return x_t

steps = 300
t = 0
img = images[index].squeeze(0)
noisy_img = p_sample(steps, img, t)

show_image(noisy_img)


# In[ ]:




