import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def sinusoidal_embeddings(pos, dim, max_pos=10000):
    """
    Compute sinusoidal position encoding.
    :param pos: Position index (array-like)
    :param dim: Encoding dimension
    :param max_pos: Scaling factor for positional encoding
    :return: Sinusoidal position encoding
    """
    position = np.expand_dims(pos, 1)  # Shape: (N, 1)
    div_term = np.exp(-np.arange(0, dim, 2) * (np.log(max_pos) / dim))
    pe = np.zeros((len(pos), dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def rope_2d(w, output_e_dim):
    """2D-RoPE
    :param img_size: Size of the input image (assume square)
    :return: 2D positional encoding with shape (1, w^2, output_e_dim)
    """
    pos = np.arange(0, w ** 2, dtype='float32')
    pos1, pos2 = pos // w, pos % w  # Row and column indices
    pos_dim = int(output_e_dim / 2)
    # Compute sinusoidal embeddings for rows and columns
    pos1 = sinusoidal_embeddings(pos1, pos_dim, 1000)
    pos2 = sinusoidal_embeddings(pos2, pos_dim, 1000)

    # Concatenate row and column embeddings
    pos_encoding = np.concatenate([pos1, pos2], axis=1)
    return pos_encoding

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    - output_e_dim : number of embeddings to output
    _____________________________________________
    """

    def __init__(self, args, n_e, e_dim, beta, n_min, output_e_dim=256, latent_w=16):
        super(VectorQuantizer, self).__init__()
        self.args = args
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.output_e_dim = output_e_dim
        self.n_min = n_min
        self.latent_w = latent_w
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        pos_embedding = rope_2d(latent_w, self.e_dim)
        if args.use_pos_embedding and not args.learnable_pos_emb:
            self.register_buffer(
                'pos_embedding',
                torch.tensor(pos_embedding, dtype=torch.float32)
            )
    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances d from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # sort distances and get indices of the smallest output_e_dim distances
        sorted_indices = torch.argsort(d, dim=1)[:, :self.output_e_dim]

        # generate new one-hot encodings for the smallest output_e_dim distances
        min_encodings = torch.zeros(z_flattened.shape[0], self.n_e).to(z)
        min_encodings = min_encodings.scatter(1, sorted_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight)

        if self.args.use_pos_embedding and not self.args.learnable_pos_emb:
            z_q = z_q.view(z.shape[0], self.latent_w**2, self.e_dim)
            z_q = z_q + self.pos_embedding

        z_q = z_q.view(z.shape)


        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, (perplexity, min_encodings, sorted_indices, d), self.embedding.weight

    def inference_cat(self, z):
        """
        Implement the selection of the first n_min smallest distances and ensure no duplicate classifications.
        Input shape: z.shape = (batch_size, channel, height, width)
        Output shape: z_q.shape = (batch_size, n_min, channel)
        """
        batch_size, _, height, width = z.shape

        # Rearrange z to (batch_size, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()  # (batch_size, height, width, channel)
        z_flattened = z.view(batch_size, -1, self.e_dim)  # (batch_size, h*w, channel)

        # Calculate distances d
        d = torch.sum(z_flattened ** 2, dim=2, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())  # (batch_size, h*w, n_e)

        # Reshape d to (batch_size, h*w, n_e)
        d = d.view(batch_size, -1, self.n_e)

        # Initialize one-hot encoding matrix
        min_encodings = torch.zeros(batch_size, self.n_min, self.n_e).to(z)
        pos_embeddings = torch.zeros(batch_size, self.n_min, self.latent_w**2).to(z)
        # Perform non-repetitive selection for each image's (h*w, n_e) range
        for i in range(batch_size):
            # seen = set()

            min_values, min_indices = torch.min(d, dim=2) 

            sorted_indices = torch.zeros(d.shape[1], 2, device=z.device)  

            sorted_indices[:, 0] = min_indices.view(-1)  
            sorted_indices[:, 1] = min_values.view(-1) 

            _, sorted_order = torch.sort(sorted_indices[:, 1], dim=0)
            position_index = torch.arange(0,self.latent_w**2).to(z.device, dtype=torch.int64)
            position_index = position_index[sorted_order][:self.n_min]
            sorted_indices = sorted_indices[sorted_order]
            indices_tensor = sorted_indices[:self.n_min,:1].to(z.device, dtype=torch.int64)

            min_encodings[i].scatter_(1, indices_tensor, 1)
            pos_embeddings[i].scatter_(1, position_index.unsqueeze(1), 1)
            
        # Get the quantized latent vectors and reshape to (batch_size, n_min, channel)
        z_q = torch.matmul(min_encodings.view(-1, self.n_e), self.embedding.weight)  # (batch_size * h * w, channel)
        if self.args.use_pos_embedding:
            assert not self.args.learnable_pos_emb,"without attention pool, pos embedding shouldn't be learnable"
            pos_z_q = torch.matmul(pos_embeddings.view(-1, self.latent_w**2), self.pos_embedding)
            z_q = z_q + pos_z_q
        z_q = z_q.view(batch_size, -1, self.e_dim)  # (batch_size, n_min, channel)

        return z_q

    def inference(self, z):
        """
        Implement the selection of the first n_min smallest distances and ensure no duplicate classifications.
        Input shape: z.shape = (batch_size, channel, height, width)
        Output shape: z_q.shape = (batch_size, n_min, channel)
        """
        batch_size, _, height, width = z.shape

        # Rearrange z to (batch_size, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()  # (batch_size, height, width, channel)
        z_flattened = z.view(batch_size, -1, self.e_dim)  # (batch_size, h*w, channel)

        # Calculate distances d
        d = torch.sum(z_flattened ** 2, dim=2, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())  # (batch_size, h*w, n_e)

        # Reshape d to (batch_size, h*w, n_e)
        d = d.view(batch_size, -1, self.n_e)

        # Initialize one-hot encoding matrix
        min_encodings = torch.zeros(batch_size, self.n_min, self.n_e).to(z)

        # Perform non-repetitive selection for each image's (h*w, n_e) range
        for i in range(batch_size):
            seen = set()
            selected_indices = []
            min_values, min_indices = torch.min(d, dim=2)  
            sorted_indices = torch.zeros(d.shape[1], 2, device=z.device)  
            sorted_indices[:, 0] = min_indices.view(-1)  
            sorted_indices[:, 1] = min_values.view(-1)  
            # Sort distances for the current pixel

            count = 0
            _, sorted_order = torch.sort(sorted_indices[:, 1], dim=0)
            sorted_indices = sorted_indices[sorted_order]
            for idx in sorted_indices[:, 0]:
                if idx.item() not in seen:
                    selected_indices.append(idx.item())
                    seen.add(idx.item())
                    count += 1
                if count >= self.n_min:
                    break
            # Update one-hot encoding matrix
            indices_tensor = torch.tensor(selected_indices, dtype=torch.int64).to(z)  # Convert to tensor
            min_encodings[i, :len(selected_indices)].scatter_(1, indices_tensor.unsqueeze(1).to(torch.int64), 1)

        # Get the quantized latent vectors and reshape to (batch_size, n_min, channel)
        z_q = torch.matmul(min_encodings.view(-1, self.n_e), self.embedding.weight)  # (batch_size * h * w, channel)
        z_q = z_q.view(batch_size, -1, self.e_dim)  # (batch_size, n_min, channel)

        return z_q

    def inference_time(self, z):
        """
        Implement the selection of the first n_min smallest distances and ensure no duplicate classifications.
        Input shape: z.shape = (batch_size, channel, height, width)
        Output shape: z_q.shape = (batch_size, n_min, channel)
        """
        timings = {}  

        batch_size, _, height, width = z.shape

        # Step 1: Rearrange z and flatten
        start_time = time.perf_counter()
        z = z.permute(0, 2, 3, 1).contiguous()  # (batch_size, height, width, channel)
        z_flattened = z.view(batch_size, -1, self.e_dim)  # (batch_size, h*w, channel)
        end_time = time.perf_counter()
        timings['Rearrange and flatten'] = (end_time - start_time) * 1000 

        # Step 2: Calculate distances d
        start_time = time.perf_counter()
        d = torch.sum(z_flattened ** 2, dim=2, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())  # (batch_size, h*w, n_e)
        end_time = time.perf_counter()
        timings['Calculate distances'] = (end_time - start_time) * 1000  

        # Step 3: Reshape d
        start_time = time.perf_counter()
        d = d.view(batch_size, -1, self.n_e)
        end_time = time.perf_counter()
        timings['Reshape distances'] = (end_time - start_time) * 1000  

        # Step 4: Initialize one-hot encoding matrix
        start_time = time.perf_counter()
        min_encodings = torch.zeros(batch_size, self.n_min, self.n_e).to(z)
        end_time = time.perf_counter()
        timings['Initialize one-hot encoding matrix'] = (end_time - start_time) * 1000  # 转换为毫秒

        # Step 5: Non-repetitive selection
        start_time = time.perf_counter()
        for i in range(batch_size):
            seen = set()
            selected_indices = []
            min_values, min_indices = torch.min(d, dim=2) 
            sorted_indices = torch.zeros(d.shape[1], 2, device=z.device) 
            sorted_indices[:, 0] = min_indices.view(-1)
            sorted_indices[:, 1] = min_values.view(-1) 
            # Sort distances for the current pixel

            count = 0
            _, sorted_order = torch.sort(sorted_indices[:, 1], dim=0)
            sorted_indices = sorted_indices[sorted_order]
            for idx in sorted_indices[:, 0]:
                if idx.item() not in seen:
                    selected_indices.append(idx.item())
                    seen.add(idx.item())
                    count += 1
                if count >= self.n_min:
                    break

            # Update one-hot encoding matrix
            indices_tensor = torch.tensor(selected_indices, dtype=torch.int64).to(z)  # Convert to tensor
            min_encodings[i, :len(selected_indices)].scatter_(1, indices_tensor.unsqueeze(1).to(torch.int64), 1)
        end_time = time.perf_counter()
        timings['Non-repetitive selection'] = (end_time - start_time) * 1000  

        # Step 6: Get quantized latent vectors
        start_time = time.perf_counter()
        z_q = torch.matmul(min_encodings.view(-1, self.n_e), self.embedding.weight)  # (batch_size * h * w, channel)
        z_q = z_q.view(batch_size, -1, self.e_dim)  # (batch_size, n_min, channel)
        end_time = time.perf_counter()
        timings['Quantized latent vectors'] = (end_time - start_time) * 1000  

        print("timing (ms)：", timings)
        return z_q

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert (in_channels % head_size == 0), 'The size of head should be divided by the number of channels.'

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.num = 0

    def forward(self, x, y=None):
        h_ = x
        h_ = self.norm1(h_)
        if y is None:
            y = h_
        else:
            y = self.norm2(y)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, self.head_size, self.att_size, h * w)
        q = q.permute(0, 3, 1, 2)  # b, hw, head, att

        k = k.reshape(b, self.head_size, self.att_size, h * w)
        k = k.permute(0, 3, 1, 2)

        v = v.reshape(b, self.head_size, self.att_size, h * w)
        v = v.permute(0, 3, 1, 2)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        scale = int(self.att_size) ** (-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k)
        w_ = F.softmax(w_, dim=3)
        atten_weight = copy.deepcopy(w_)

        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous()  # [b, h*w, head, att]


        w_ = w_.view(b, h, w, -1)
        w_ = w_.permute(0, 3, 1, 2)

        w_ = self.proj_out(w_)

        return x + w_, atten_weight


class MultiHeadEncoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, double_z=True, enable_mid=True,
                 head_size=1, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.enable_mid = enable_mid

        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        hs = {}
        # timestep embedding
        temb = None
        # downsampling
        h = self.conv_in(x)
        hs['in'] = h
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h,_ = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                # hs.append(h)
                hs['block_' + str(i_level)] = h
                h = self.down[i_level].downsample(h)
        # middle
        # h = hs[-1]
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            hs['block_' + str(i_level) + '_atten'] = h
            h,atten_weight = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)
            hs['mid_atten'] = h

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # hs.append(h)
        hs['out'] = h
        return hs, atten_weight


class MultiHeadDecoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):

        # import pdb
        # pdb.set_trace()

        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class MultiHeadDecoderTransformer(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0, resamp_with_conv=True, in_channels=3,
                 resolution=512, z_channels=256, give_pre_end=False, enable_mid=True,
                 head_size=1, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.enable_mid = enable_mid

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        if self.enable_mid:
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)
            self.mid.attn_1 = MultiHeadAttnBlock(block_in, head_size)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in, head_size))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs):
        # assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape

        # import pdb
        # pdb.set_trace()
        # z = quant feature
        # hs = LQ feature

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.enable_mid:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h, hs['mid_atten'])
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, hs['block_' + str(i_level) + '_atten'])
                    # hfeature = h.clone()
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQVAEencoder(nn.Module):
    def __init__(self, args, n_embed=1024, embed_dim=512, ch=64, out_ch=3, ch_mult=(1,2,2,2,4,8),
                 num_res_blocks=2, attn_resolutions=[16], dropout=0.0, in_channels=3,
                 resolution=512, z_channels=512, double_z=False, enable_mid=True,
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1, n_min=77,
                 **ignore_kwargs):
        super(VQVAEencoder, self).__init__()
        self.args = args
        self.cat_prompt_embedding = args.cat_prompt_embedding
        self.use_att_pool = args.use_att_pool
        self.encoder = MultiHeadEncoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                                        attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                                        resolution=resolution, z_channels=z_channels, double_z=double_z,
                                        enable_mid=enable_mid, head_size=head_size)
        self.latent_w = int(resolution/(2**(len(ch_mult)-1)))

        self.quantize = VectorQuantizer(args, n_embed, embed_dim, beta=0.25, n_min=n_min, latent_w=self.latent_w)

        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)


    def encode(self, x):
        hs, atten_weight = self.encoder(x)
        h = self.quant_conv(hs['out'])
        if self.use_att_pool:
            quant = self.quantize(h)[0]
            quant = quant.permute(0, 2, 3, 1)
            quant = quant.view(quant.shape[0], -1, quant.shape[-1])

        else:
            if self.cat_prompt_embedding:
                quant = self.quantize.inference_cat(h)
            else:
                quant = self.quantize.inference(h)
        return quant

    def forward(self, input):
        quant = self.encode(input)

        return quant
