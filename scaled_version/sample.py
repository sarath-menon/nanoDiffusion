import torch as th
from torchvision.utils import save_image
from download import find_model
from dit import DiT, DiTConfig, load_pretrained_model
from diffusers.models import AutoencoderKL
import os
import numpy as np
from tqdm.auto import tqdm
import numpy as np
from scheduler import GaussianDiffusion 

th.manual_seed(1)
device = "cuda" if th.cuda.is_available() else "cpu"

diffusion_steps = 1000
n_sampling_steps = 5
cfg_scale = 4.0
class_labels = [11] # Labels to condition the model with (feel free to change)

@th.no_grad()
def p_sample_loop(model_output, x, t, gd):
    # safety checks
    B, C = x.shape[:2]
    assert model_output.shape == (B, C * 2, *x.shape[2:])

    # get model output
    noise_pred, model_var_values = th.split(model_output, C, dim=1)
   
    # mean prediction
    coeff1 = th.sqrt(1./gd.alphas) 
    coeff2 = gd.betas * th.sqrt(1./gd.alphas) * th.sqrt(1./(1-gd.alpha_prod))
    mean_pred = coeff1[t] * x - coeff2[t] * noise_pred

    # var prediction
    min_log = th.full_like(x, gd.posterior_log_var[t])
    max_log = th.full_like(x, th.log(gd.betas[t]))
    frac = (model_var_values + 1) / 2
    model_log_var = frac * max_log + (1 - frac) * min_log
    std_dev_pred = th.exp(0.5 * model_log_var)

    # inference
    nonzero_mask = 0. if t == 0 else 1.; 
    noise = th.randn_like(x)
    x_prev = mean_pred + nonzero_mask * std_dev_pred * noise

    return x_prev 

def inference(x,y):

    # create params for gaussian diffusion
    indices = list(range(n_sampling_steps))[::-1]
    indices = tqdm(indices) # for progres bar
   
    gd = GaussianDiffusion(diffusion_steps, n_sampling_steps, sampling=True, device=device)

    for i in indices:
        t = th.tensor([gd.sampling_ts[i]] * x.shape[0], device=device) 
        model_output = model.forward_with_cfg(x, t, y, cfg_scale)

        x = p_sample_loop(model_output, x,  i, gd) 
    return x

# setup diffusion transformer
dit_cfg = DiTConfig()
model = DiT(dit_cfg)
model = load_pretrained_model(model)
model.eval()  

# diffusion = create_diffusion(str(n_sampling_steps))
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
# decoder = StableDiffusionDecoder()

 # Convert image class to noise latent:
latent_size = dit_cfg.input_size
n = len(class_labels)
z = th.randn(n, 4, latent_size, latent_size, device=device)
y = th.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = th.cat([z, z], 0)
y_null = th.tensor([dit_cfg.num_classes] * n, device=device)
y = th.cat([y, y_null], 0)

samples = inference(z,y)

# convert image latent to image
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
# samples = decoder(samples)
samples = vae.decode(samples / 0.18215).sample
# print(samples.shape)


# Save and display images:
path = os.getcwd()
output_dir = os.path.join(path, "out")
save_image(samples, os.path.join(output_dir, "sample.png"), nrow=1, normalize=True, value_range=(-1, 1))

