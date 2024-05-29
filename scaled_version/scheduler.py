import torch as th
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

class GaussianDiffusion:
    def __init__(self, diffusion_steps, n_sampling_steps, sampling=False, device='cpu'):
        self.diffusion_steps = diffusion_steps
        self.n_sampling_steps = n_sampling_steps
        self.device = device

        self.betas = self.linear_beta_schedule(diffusion_steps).float().to(device)

        if sampling:
            self.sampling_ts = self.get_sampling_ts_from_diffusion_ts()
            self.betas, _ = self.get_sampling_schedule()
            
        self.alphas = 1. - self.betas
        self.alpha_prod = th.cumprod(self.alphas, 0)
        self.alpha_prod_prev = th.cat([th.tensor([1.0]), self.alpha_prod[:-1].to(self.device)])
        self.posterior_var = self.betas * (1. - self.alpha_prod_prev) / (1. - self.alpha_prod)

        if len(self.posterior_var) > 1:
            self.posterior_log_var = th.log(th.cat([self.posterior_var[1].unsqueeze(0), self.posterior_var[1:]]))
        else:
            self.posterior_log_var = th.tensor([], device=self.device)
    
    def linear_beta_schedule(self, diffusion_timesteps):
        scale = 1
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return th.linspace(beta_start, beta_end, diffusion_timesteps) 
    
    # def cosine_beta_schedule(self, timesteps, s=0.008):
    #     steps = timesteps + 1
    #     x = th.linspace(0, timesteps, steps)
    #     alphas_cumprod = th.cos(((x / timesteps) + s) / (1 + s) * th.pi * 0.5) ** 2
    #     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    #     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    #     return th.clip(betas, 0.0001, 0.9999)


    # get evenly spaced time steps for the sampling schedule
    def get_sampling_ts_from_diffusion_ts(self):
        section_counts = [self.n_sampling_steps]
        size_per = self.diffusion_steps // len(section_counts)
        extra = self.diffusion_steps % len(section_counts)
        start_idx = 0
        all_steps = []
        for i, section_count in enumerate(section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(f"cannot divide section of {size} steps into {section_count}")
            stride = (size - 1) / (section_count - 1) if section_count > 1 else 0
            steps = [start_idx + round(stride * j) for j in range(section_count)]
            all_steps.extend(steps)
            start_idx += size
        return all_steps

   
    def get_sampling_schedule(self):
        last_alpha_prod = 1.0
        alphas = 1. - self.betas
        alpha_prod = th.cumprod(alphas, 0)
        new_betas = []
        timestep_map = []
        for i, alpha_prod in enumerate(alpha_prod):
            if i in self.sampling_ts:
                new_betas.append(1 - alpha_prod / last_alpha_prod)
                last_alpha_prod = alpha_prod
                timestep_map.append(i)
        return th.tensor(new_betas), timestep_map

