<img src="./vq.png" width="500px"></img>

## Vector Quantization - Pytorch

A vector quantization library originally transcribed from Deepmind's tensorflow implementation, made conveniently into a package. It uses exponential moving averages to update the dictionary.

VQ has been successfully used by Deepmind and OpenAI for high quality generation of images (VQ-VAE-2) and music (Jukebox).

## Install

```bash
$ pip install vector-quantize-pytorch
```

## Usage

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 512,     # codebook size
    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
    commitment_weight = 1.   # the weight on the commitment loss
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x) # (1, 1024, 256), (1, 1024), (1)
```

## Residual VQ

This <a href="https://arxiv.org/abs/2107.03312">paper</a> proposes to use multiple vector quantizers to recursively quantize the residuals of the waveform. You can use this with the `ResidualVQ` class and one extra initialization parameter.

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 256,
    num_quantizers = 8,      # specify number of quantizers
    codebook_size = 1024,    # codebook size
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = residual_vq(x)

# (1, 1024, 256), (1, 1024, 8), (1, 8)
# (batch, seq, dim), (batch, seq, quantizer), (batch, quantizer)
```

Furthermore, <a href="https://arxiv.org/abs/2203.01941">this paper</a> uses Residual-VQ to construct the RQ-VAE, for generating high resolution images with more compressed codes.

They make two modifications. The first is to share the codebook across all quantizers. The second is to stochastically sample the codes rather than always taking the closest match. You can use both of these features with two extra keyword arguments.

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 256,
    num_quantizers = 8,
    codebook_size = 1024,
    sample_codebook_temp = 0.1, # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
    shared_codebook = True      # whether to share the codebooks for all quantizers or not
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = residual_vq(x)

# (1, 1024, 256), (8, 1, 1024), (8, 1)
# (batch, seq, dim), (quantizer, batch, seq), (quantizer, batch)
```

## Initialization

The SoundStream paper proposes that the codebook should be initialized by the kmeans centroids of the first batch. You can easily turn on this feature with one flag `kmeans_init = True`, for either `VectorQuantize` or `ResidualVQ` class

```python
import torch
from vector_quantize_pytorch import ResidualVQ

residual_vq = ResidualVQ(
    dim = 256,
    codebook_size = 256,
    num_quantizers = 4,
    kmeans_init = True,   # set to True
    kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = residual_vq(x)
```

## Increasing codebook usage

This repository will contain a few techniques from various papers to combat "dead" codebook entries, which is a common problem when using vector quantizers.

### Lower codebook dimension

The <a href="https://openreview.net/forum?id=pfNyExj7z2">Improved VQGAN paper</a> proposes to have the codebook kept in a lower dimension. The encoder values are projected down before being projected back to high dimensional after quantization. You can set this with the `codebook_dim` hyperparameter.

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    codebook_dim = 16      # paper proposes setting this to 32 or as low as 8 to increase codebook usage
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```

### Cosine similarity

The <a href="https://openreview.net/forum?id=pfNyExj7z2">Improved VQGAN paper</a> also proposes to l2 normalize the codes and the encoded vectors, which boils down to using cosine similarity for the distance. They claim enforcing the vectors on a sphere leads to improvements in code usage and downstream reconstruction. You can turn this on by setting `use_cosine_sim = True`

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    use_cosine_sim = True   # set this to True
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```

### Expiring stale codes

Finally, the SoundStream paper has a scheme where they replace codes that have hits below a certain threshold with randomly selected vector from the current batch. You can set this threshold with `threshold_ema_dead_code` keyword.

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 512,
    threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2
)

x = torch.randn(1, 1024, 256)
quantized, indices, commit_loss = vq(x)
```

### Orthogonal regularization loss

VQ-VAE / VQ-GAN is quickly gaining popularity. A <a href="https://arxiv.org/abs/2112.00384">recent paper</a> proposes that when using vector quantization on images, enforcing the codebook to be orthogonal leads to translation equivariance of the discretized codes, leading to large improvements in downstream text to image generation tasks.

You can use this feature by simply setting the `orthogonal_reg_weight` to be greater than `0`, in which case the orthogonal regularization will be added to the auxiliary loss outputted by the module.

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_size = 256,
    accept_image_fmap = True,                   # set this true to be able to pass in an image feature map
    orthogonal_reg_weight = 10,                 # in paper, they recommended a value of 10
    orthogonal_reg_max_codes = 128,             # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
    orthogonal_reg_active_codes_only = False    # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
)

img_fmap = torch.randn(1, 256, 32, 32)
quantized, indices, loss = vq(img_fmap) # (1, 256, 32, 32), (1, 32, 32), (1,)

# loss now contains the orthogonal regularization loss with the weight as assigned
```

### Multi-headed VQ

There has been a number of papers that proposes variants of discrete latent representations with a multi-headed approach (multiple codes per feature). I have decided to offer one variant where the same codebook is used to vector quantize across the input dimension `head` times.

You can also use a more proven approach (memcodes) from <a href="https://github.com/lucidrains/nwt-pytorch">NWT paper</a>

```python
import torch
from vector_quantize_pytorch import VectorQuantize

vq = VectorQuantize(
    dim = 256,
    codebook_dim = 32,         # a number of papers have shown smaller codebook dimension to be acceptable
    heads = 8,                 # number of heads to vector quantize, codebook shared across all heads
    codebook_size = 8196,
    accept_image_fmap = True
)

img_fmap = torch.randn(1, 256, 32, 32)
quantized, indices, loss = vq(img_fmap) # (1, 256, 32, 32), (1, 32, 32, 8), (1,)

# indices shape - (batch, height, width, heads)
# loss now contains the orthogonal regularization loss with the weight as assigned
```

### DDP

This repository also supports synchronizing the codebooks in a distributed settings. Below should be a working script, and also shows which flag you need to enable for it to work as expected.

```python
import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize

# ddp related imports

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def start(rank, world_size):
    setup(rank, world_size)

    net = nn.Sequential(
        nn.Conv2d(256, 256, 1),
        VectorQuantize(
            dim = 256,
            codebook_size = 1024,
            accept_image_fmap = True,
            sync_codebook = True           # this needs to be set to True
        )
    ).cuda(rank)

    ddp_mp_model = DDP(net, device_ids = [rank])
    img_fmap = torch.randn(1, 256, 32, 32).cuda(rank)
    quantized, indices, loss = ddp_mp_model(img_fmap)

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"requires at least 2 GPUs to run, but got {n_gpus}"
    mp.spawn(start, args=(world_size,), nprocs=world_size, join=True)

```

## Todo

- [x] allow for multi-headed codebooks
- [ ] support masking


## Citations

```bibtex
@misc{oord2018neural,
    title   = {Neural Discrete Representation Learning},
    author  = {Aaron van den Oord and Oriol Vinyals and Koray Kavukcuoglu},
    year    = {2018},
    eprint  = {1711.00937},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{zeghidour2021soundstream,
    title   = {SoundStream: An End-to-End Neural Audio Codec},
    author  = {Neil Zeghidour and Alejandro Luebs and Ahmed Omran and Jan Skoglund and Marco Tagliasacchi},
    year    = {2021},
    eprint  = {2107.03312},
    archivePrefix = {arXiv},
    primaryClass = {cs.SD}
}
```

```bibtex
@inproceedings{anonymous2022vectorquantized,
    title   = {Vector-quantized Image Modeling with Improved {VQGAN}},
    author  = {Anonymous},
    booktitle = {Submitted to The Tenth International Conference on Learning Representations },
    year    = {2022},
    url     = {https://openreview.net/forum?id=pfNyExj7z2},
    note    = {under review}
}
```

```bibtex
@misc{shin2021translationequivariant,
    title   = {Translation-equivariant Image Quantizer for Bi-directional Image-Text Generation}, 
    author  = {Woncheol Shin and Gyubok Lee and Jiyoung Lee and Joonseok Lee and Edward Choi},
    year    = {2021},
    eprint  = {2112.00384},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@unknown{unknown,
    author  = {Lee, Doyup and Kim, Chiheon and Kim, Saehoon and Cho, Minsu and Han, Wook-Shin},
    year    = {2022},
    month   = {03},
    title   = {Autoregressive Image Generation using Residual Quantization}
}
```