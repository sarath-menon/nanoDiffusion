import torch as th
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from dataclasses import dataclass
from model import DiT, DiTConfig, GaussianDiffusion 
th.manual_seed(42)

@dataclass
class TrainConfig:
    image_size: int = 28 # fashion mnist img size
    patch_size = 14
    num_epochs: int = 3
    eval_iters: int = 200
    eval_interval: int = 100
    batch_size: int = 128
    diffusion_steps = 300
    lr: float = 1e-3

train_cfg = TrainConfig()
device = "cuda" if th.cuda.is_available() else "cpu"
print("Using device", device)

# setup diffusion transformer
dit_cfg = DiTConfig()
model = DiT(dit_cfg)
# model.load_state_dict(th.load("weights/simple_fmnist_weights.pth"))
model.train()  # important! 

gd = GaussianDiffusion(device=device)
optimizer = th.optim.AdamW(model.parameters(), lr=train_cfg.lr)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
])

train_dataset  = datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform)
trainloader = th.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size,shuffle=True)

def p_loss(model, x_start, t, y):
    # sample an image
    x_t, noise = gd.p_sample(x_start, t)

    # get predicted noise
    B, C = x_t.shape[:2]
    model_output = model(x_t, t, y)
    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
    noise_pred, model_var_values = th.split(model_output, C, dim=1)

    return th.nn.functional.mse_loss(noise, noise_pred)

for epoch in range(train_cfg.num_epochs):
    running_loss = 0.0
    print("Starting epoch")

    for i, (x, y) in enumerate(trainloader):
        x = x.to(device) 
        y = y.to(device)

        # Generate random timesteps for each image latent
        t = th.randint(0, train_cfg.diffusion_steps, (x.size(0),), device=device)

        loss = p_loss(model, x, t, y)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

         # print statistics
        running_loss += loss.item()
        if i!=0 and i % train_cfg.eval_interval == 0:    
            print(f'[{epoch + 1}, {i + 1:5d}] runnning loss: {running_loss / train_cfg.eval_interval:.3f}')
            running_loss = 0.0
            th.save(model.state_dict(), 'weights/simple_fmnist_weights.pth')



