import torch as th
from torchvision.utils import save_image
from model import DiT, DiTConfig, GaussianDiffusion 
import os
import numpy as np
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

th.manual_seed(1)
device = "cuda" if th.cuda.is_available() else "cpu"

class_labels = [0] # Labels to condition the model with (feel free to change)
cfg_scale = 4.0
save_interval = 50
diffusion_steps = 300

@th.no_grad()
def p_sample_loop(model_output, x, t, gd):
    # safety checks
    B, C = x.shape[:2]
    assert model_output.shape == (B, C * 2, *x.shape[2:])

    # get model output
    noise_pred, model_var_values = th.split(model_output, C, dim=1)

    # mean prediction
    a = th.sqrt(1./gd.alphas) 
    b = gd.betas * th.sqrt(1./gd.alphas) * th.sqrt(1./(1-gd.alpha_prod))
    mean_pred = a[t] * x - b[t] * noise_pred

    # fix var = beta_t
    std_dev_fixed = th.sqrt(gd.posterior_var[t])

    # inference
    noise = th.randn_like(x)
    x_prev = mean_pred +  std_dev_fixed * noise

    return x_prev 

def inference(x,y, diffusion_steps):

    # create params for gaussian diffusion
    indices = list(range((diffusion_steps)))[::-1]
    indices = tqdm(indices) # for progres bar
   
    gd = GaussianDiffusion(device=device)
    x_hist = []

    for i in indices:
        t = th.tensor([i] * x.shape[0], device=device) 
        model_output = model.forward_with_cfg(x, t, y, cfg_scale)
        x = p_sample_loop(model_output, x,  i, gd) 

        if i % save_interval == 0:
            x_hist.append(x)
    return x, x_hist

# setup diffusion transformer
dit_cfg = DiTConfig()
model = DiT(dit_cfg)
model.load_state_dict(th.load("weights/simple_fmnist_weights.pth"))
model.eval()  

 # Convert image class to noise latent:
latent_size = dit_cfg.input_size
n = len(class_labels)
z = th.randn(n, 1, latent_size, latent_size, device=device)
y = th.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = th.cat([z, z], 0)
y_null = th.tensor([dit_cfg.num_classes] * n, device=device)
y = th.cat([y, y_null], 0)

samples, x_hist = inference(z,y, diffusion_steps)

# plotting and saving the images
for i, x in enumerate(x_hist):
    # convert image latent to image
    x, _ = x.chunk(2, dim=0)  # Remove null class samples
    x= (x * 0.5) + 0.5 # # invert the normalization
    x_hist[i] = x.squeeze().cpu().numpy() 

fig, axes = plt.subplots(1, len(x_hist), figsize=(len(x_hist) * 2, 2))
for ax, img in zip(axes, x_hist):
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig(os.path.join(os.getcwd(), "out", "sample_fmnist"), bbox_inches='tight')
plt.close()