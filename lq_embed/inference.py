import os
import time
current_dir = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqvae import VQVAEencoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads."

        self.query_fc = nn.Linear(embed_size, embed_size)
        self.key_fc = nn.Linear(embed_size, embed_size)
        self.value_fc = nn.Linear(embed_size, embed_size)

        self.out_fc = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)

        # Scaled dot-product attention
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)
        self.scale = self.scale.to(query.device)
        query = self.query_fc(query)  # (batch_size, query_len, embed_size)
        key = self.key_fc(key)  # (batch_size, key_len, embed_size)
        value = self.value_fc(value)  # (batch_size, value_len, embed_size)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, query_len, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, key_len, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, value_len, head_dim)

        energy = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)  # (batch_size, num_heads, query_len, key_len)

        attention = self.dropout(attention)

        out = torch.matmul(attention, value)  # (batch_size, num_heads, query_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  
        # (batch_size, query_len, embed_size)

        out = self.out_fc(out)  # (batch_size, query_len, embed_size)

        return out

class AttentionPoolingLayer(nn.Module):
    def __init__(self, args, input_n, input_dim, num_latent_queries, num_heads):
        super(AttentionPoolingLayer, self).__init__()
        self.args = args
        self.num_latent_queries = num_latent_queries
        self.latent_queries = nn.Parameter(torch.randn(num_latent_queries, input_dim))
        self.multihead_attn = MultiHeadAttention(embed_size=input_dim, num_heads=num_heads)
        if self.args.use_pos_embedding and self.args.learnable_pos_emb:
            self.pos_embedding = nn.Parameter(torch.randn(input_n, input_dim))
    def forward(self, x):
        # x shape: (B, N, C)
        B, N, C = x.shape
        if self.args.use_pos_embedding and self.args.learnable_pos_emb:
            pos_embedding = self.pos_embedding.unsqueeze(0).expand(B, -1, -1).to(x.device)
            x = pos_embedding + x
        latent_queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1).to(x.device)
        # Multihead Attention

        attn_output = self.multihead_attn(latent_queries, x, x)

        return attn_output


class vqvae_encoder(nn.Module):
    def __init__(self, args, n_embed=1024, embed_dim=512, ch=64, out_ch=3, ch_mult=(1,2,2,2,4,8),
                 num_res_blocks=2, attn_resolutions=[16], dropout=0.0, in_channels=3,
                 resolution=512, z_channels=512, double_z=False, enable_mid=True,
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, n_min=77):
        super(vqvae_encoder, self).__init__()
        self.args = args
        if args.cat_prompt_embedding:
            n_min = n_min*2
        self.encoder = VQVAEencoder(args=args, n_embed=n_embed, embed_dim=embed_dim, ch=ch, out_ch=out_ch, ch_mult=ch_mult,
                 num_res_blocks=num_res_blocks, attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                 resolution=resolution, z_channels=z_channels, double_z=double_z, enable_mid=enable_mid,
                 fix_decoder=fix_decoder, fix_codebook=fix_codebook, fix_encoder=fix_encoder, head_size=head_size, n_min=n_min)
        self.encoder.requires_grad_(False)
        state_dict = torch.load(args.img_encoder_weight, weights_only=False, map_location="cpu")["state_dict"]
        new_state_dict = {}

        for key in state_dict:
            if key.startswith('vqvae_LQ.') and not key.startswith('vqvae_LQ.decoder'):
                new_key = key.replace('vqvae_LQ.', '')
                new_state_dict[new_key] = state_dict[key]
        self.encoder.load_state_dict(new_state_dict, strict=False)

    @torch.no_grad()
    def forward(self, img):
        self.encoder.eval()
        img = F.interpolate(img, size=(512, 512), mode='bilinear', align_corners=False)

        img = (img - 0.5) / 0.5

        feat = self.encoder(img)

        if self.args.cat_prompt_embedding and not self.args.use_att_pool:
            b, n, c = feat.shape
            assert n % 2 == 0, "The n dimension must be divisible by 2"

            # Split tensor into two halves along the n dimension
            first_half, second_half = torch.split(feat, n // 2, dim=1)  # Shape: (1, n/2, c)

            # Concatenate the two halves along the c dimension
            feat = torch.cat([first_half, second_half], dim=2)  # Shape: (1, n/2, 2c)

        return feat

    def encode(self, x):
        timings = {}

        start_time = time.perf_counter()
        hs, atten_weight = self.encoder.encoder(x)
        end_time = time.perf_counter()
        timings['encoder'] = (end_time - start_time) * 1000  

        start_time = time.perf_counter()
        h = self.encoder.quant_conv(hs['out'])
        end_time = time.perf_counter()
        timings['quant_conv'] = (end_time - start_time) * 1000  

        start_time = time.perf_counter()
        quant = self.encoder.quantize.inference_time(h)
        end_time = time.perf_counter()
        timings['quantize'] = (end_time - start_time) * 1000  

        return quant, timings


class AttentionPooling(AttentionPoolingLayer):
    def __init__(self, args, input_n, input_dim, num_latent_queries, num_heads=1):
        super(AttentionPooling, self).__init__(args, input_n, input_dim, num_latent_queries, num_heads)
        self.args = args
        self.attention_pool = AttentionPoolingLayer(args, input_n, input_dim, num_latent_queries, num_heads)
        self.attention_pool.requires_grad_(True)

    def forward(self, x):
        x = self.attention_pool(x)
        if self.args.cat_prompt_embedding:
            b, n, c = x.shape
            assert n % 2 == 0, "The n dimension must be divisible by 2"

            # Split tensor into two halves along the n dimension
            first_half, second_half = torch.split(x, n // 2, dim=1)  # Shape: (1, n/2, c)

            # Concatenate the two halves along the c dimension
            x = torch.cat([first_half, second_half], dim=2)  # Shape: (1, n/2, 2c)
            return x

    def save(self, path):
        torch.save(self.attention_pool.state_dict(), os.path.join(path, "attn_pool"))

    def load(self, path):
        self.attention_pool.load_state_dict(torch.load(os.path.join(path, "attn_pool"), map_location="cpu"))


class TwoLayerConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoLayerConv1x1, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels // 2, out_channels, kernel_size=1)

        self.silu = nn.SiLU()

    def forward(self, x):

        x = x.permute(0, 2, 1)
        assert x.shape[2]==77

        x = self.silu(self.conv1(x))
        x = self.silu(self.conv2(x))

        x = x.permute(0, 2, 1)
        return x