import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images


class EvalTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.model.load_checkpoint(args.trans_path)
        self.model = self.model.eval()
        self.eval_trans(args)

    def eval_trans(self, args):
        train_dataset = load_data(args)
        icnt = 0
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    icnt = icnt + 1
                    imgs = imgs.to(device=args.device)
                    log, img_x, img_x_rec, img_x_sample, img_x_vqgan = self.model.log_images(imgs[0][None])
                    vutils.save_image(img_x, os.path.join("result_fig/orig", f"{icnt}.jpg"), nrow=4)
                    vutils.save_image(img_x_sample, os.path.join("result_fig/half", f"{icnt}.jpg"), nrow=4)
                    vutils.save_image(img_x_vqgan, os.path.join("result_fig/vqgan", f"{icnt}.jpg"), nrow=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=256, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='/content/drive/MyDrive/test_data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='/content/drive/MyDrive/models_checkpoints/vqgan.pt', help='Path to checkpoint.')
    parser.add_argument('--trans-path', type=str, default='/content/drive/MyDrive/models_checkpoints/transformer.pt', help='Path to trans checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=0, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    args.dataset_path = r"/content/drive/MyDrive/test_data"
    args.checkpoint_path = r"/content/drive/MyDrive/models_checkpoints/vqgan.pt"
    args.trans_path = r"/content/drive/MyDrive/models_checkpoints/transformer.pt"

    eval_transformer = EvalTransformer(args)
