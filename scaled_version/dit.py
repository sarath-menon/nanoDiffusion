
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import numpy as np
from download import find_model

@dataclass
class DiTConfig:
    input_size: int = 32 # size of image latent
    patch_size: int = 2
    reg_tokens: int = 0
    n_layers: int = 28
    n_heads: int = 16
    n_embed: int = 1152
    in_chans: int = 4 # no. channels of latent
    dropout: float = 0.0
    device: str = 'cpu'
    mlp_ratio: int = 4
    num_classes: int = 1000
    bias: bool = True
    class_dropout_prob: float = 0.1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class AdaLNModulation(nn.Module):
    def __init__(self, hidden_size, out_scale):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, out_scale * hidden_size, bias=True)

    def forward(self, x):
        x = self.silu(x)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, H, W, patch_size=16, in_chans=3, n_embed=100):
        super().__init__()

        self.num_patches = (H * W) // (patch_size ** 2)
        self.patch_size = patch_size

        # conv operation acts as a linear embedding
        self.proj = nn.Conv2d(in_chans, n_embed, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  #(B, C, H, W) -> (B, n_embed, H, W)
        x = x.flatten(2) #(B, n_embed, H, W) -> (B, n_embed, H*W)
        x = x.transpose(1, 2) # (B, n_embed, H*W) -> (B, H*W, n_embed)

        return x

class AcausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embed % cfg.n_heads == 0, "embedding dim must be divisible by n_heads"

        self.n_heads, self.n_embed = cfg.n_heads, cfg.n_embed
        self.head_size = self.n_embed // self.n_heads

        # key,query,value matrices as a single batch for efficiency
        self.qkv = nn.Linear(cfg.n_embed, 3*cfg.n_embed, bias=cfg.bias)

        # output layer
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B,T,C = x.shape

        # get q,k,v, matrices and reshape for multi-head attention
        q, k, v = self.qkv(x).chunk(3, dim=2)
        q, k, v = [z.view(B, T, self.n_heads, self.head_size).transpose(1, 2) for z in (q, k, v)]  # (B, nh, T, hs)
        
        weights = q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1))) # (B,T,hs) @ (B,hs,T) --->  (B,T,T)
        weights = F.softmax(weights, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out 

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        n_embed = cfg.n_embed

        self.attn = AcausalSelfAttention(cfg)
        self.mlp = FeedForward(cfg)

        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed) 

        self.adaLN_modulation = AdaLNModulation(n_embed, 6)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6 , dim=1)
        
        # modulate -> scale and shift layer norm output 
        # gate_msa -> scale the attention output
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_embed, dropout = cfg.n_embed, cfg.dropout

        self.fc1 = nn.Linear(n_embed, cfg.mlp_ratio * n_embed)
        self.fc2 = nn.Linear(cfg.mlp_ratio * n_embed, n_embed)  # projection layer
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 2 == 0
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first
    grid = torch.stack(grid, dim=0).reshape([2, 1, grid_size, grid_size])

    # use half of dimensions to encode grid_h and grid_w
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)

    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros([extra_tokens, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega = 1. / 10000**(omega / (embed_dim / 2.))  # (D/2,)

    out = pos.reshape(-1, 1) * omega  # (M, D/2), outer product
    emb = torch.cat([torch.sin(out), torch.cos(out) ], axis=1)  # (M, D)
    return emb

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = AdaLNModulation(hidden_size, 2)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__(self, cfg):
        super(DiT, self).__init__()

        image_height, image_width = pair(cfg.input_size)
        patch_height, patch_width = pair(cfg.patch_size)
        self.device = cfg.device
        self.out_channels = cfg.in_chans * 2
        self.patch_size = cfg.patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # patchifier
        self.patch_embedder = PatchEmbed(cfg.input_size, cfg.input_size, cfg.patch_size, cfg.in_chans, cfg.n_embed)

        # timestep embedder
        self.timestep_embedder = TimestepEmbedder(cfg.n_embed)

        # label embedded
        self.label_embedder = LabelEmbedder(cfg.num_classes, cfg.n_embed, cfg.class_dropout_prob)

        # self.reg_token = nn.Parameter(torch.zeros(1, cfg.reg_tokens, cfg.n_embed)) 
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        # position embedding
        pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.n_embed) * .02)
        print("Num patches", num_patches)

        pos_embed_tensor = get_2d_sincos_pos_embed(pos_embed.shape[-1], int(num_patches ** 0.5))
        self.pos_embed = nn.Parameter(pos_embed_tensor.float().unsqueeze(0))  # Convert to float, add batch dimension, and wrap in nn.Parameter

        self.blocks = nn.ModuleList([
             *[Block(cfg) for _ in range(cfg.n_layers)]
        ])

        self.final_layer = FinalLayer(cfg.n_embed, cfg.patch_size, self.out_channels)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # Rearrange to (N, C, H, p, W, p)
        imgs = x.view(x.shape[0], c, h * p, w * p)  # Combine patch dimensions with height and width
        return imgs

    def forward(self, x, t, y, targets=None):

        # get timestep and label embeddings
        timestep_embed = self.timestep_embedder(t)
        label_embed = self.label_embedder(y, False) # False-> not in training mode
        c = timestep_embed + label_embed

        x = self.patch_embedder(x) + self.pos_embed
        # x = self.blocks(x,c)

        for block in self.blocks:
            x = block(x, c)

        # final linear layer
        x = self.final_layer(x, c)
        x = self.unpatchify(x)   

        return x

    # Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance 
    # (https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb)
    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def load_pretrained_model(model, ckpt_path = "DiT-XL-2-256x256.pt"):
    
    state_dict = model.state_dict()

    # Auto-download a pre-trained model or load a custom DiT checkpoint 
    pretrained_state_dict = find_model(ckpt_path)

    # Define a mapping from old keys to new keys
    key_mapping = {}
    for key in pretrained_state_dict.keys():
        if ".adaLN_modulation.1.weight" in key:
            new_key = key.replace(".adaLN_modulation.1.weight", ".adaLN_modulation.linear.weight")
            key_mapping[key] = new_key
        elif ".adaLN_modulation.1.bias" in key:
            new_key = key.replace(".adaLN_modulation.1.bias", ".adaLN_modulation.linear.bias")
            key_mapping[key] = new_key
        elif "y_embedder" in key:
            new_key = key.replace("y_embedder", "label_embedder")
            key_mapping[key] = new_key
        elif "x_embedder" in key:
            new_key = key.replace("x_embedder", "patch_embedder")
            key_mapping[key] = new_key
        elif "t_embedder" in key:
            new_key = key.replace("t_embedder", "timestep_embedder")
            key_mapping[key] = new_key

    # Load state dict layer by layer:
    for name, param in pretrained_state_dict.items():
        # Map the state_dict name to the model's state_dict name if necessary
        mapped_name = key_mapping.get(name, name)
        if mapped_name in state_dict:
            try:
                state_dict[mapped_name].copy_(param)
                # print(f"Layer {mapped_name} loaded successfully")
            except Exception as e:
                print(f"Failed to load {mapped_name}. Reason: {e}")
        else:
            print(f"{mapped_name} not found in the model's state_dict")

    print("Model loaded")
    return model