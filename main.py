#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.image_size, config.magnification, config.batch_size, 
                               config.dataset, config.mode, config.num_workers)

    # Solver for training and testing our networks.
    solver = Solver(celeba_loader, config)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=4, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='output image resolution')
    parser.add_argument('--magnification', type=int, default=4, choices=[2, 4, 8, 16], help='image magnification scale')
    parser.add_argument('--tg_conv_dim', type=int, default=64, help='number of conv filters in the first layer of T_G')
    parser.add_argument('--td_conv_dim', type=int, default=64, help='number of conv filters in the first layer of T_D')
    parser.add_argument('--tg_repeat_num', type=int, default=6, help='number of residual blocks in TG')
    parser.add_argument('--td_repeat_num', type=int, default=6, help='number of strided conv layers in TD')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA'], help='dataset is CelebA')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=300000, help='number of total iterations for training T_D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--tg_lr', type=float, default=1e-4, help='learning rate for T_G')
    parser.add_argument('--td_lr', type=float, default=1e-4, help='learning rate for T_D')
    parser.add_argument('--eg_lr', type=float, default=1e-4, help='learning rate for E_G')
    parser.add_argument('--edc_lr', type=float, default=1e-4, help='learning rate for E_Dc')
    parser.add_argument('--edt_lr', type=float, default=1e-4, help='learning rate for E_Dt')
    parser.add_argument('--tv_weight', type=float, default=10, help='tv_loss weight')
    parser.add_argument('--n_critic', type=int, default=5, help='number of T_D updates per each T_G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset', 
                        default=['Male', 'Mustache', 'Big_Nose', 'Mouth_Slightly_Open'])
    # default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    # default=['Male', 'Young', 'Bags_Under_Eyes', 'Heavy_Makeup', 'No_Beard', '5_o_Clock_Shadow', 'Mustache', 'Pointy_Nose', 'Big_Nose', 'Blurry', 'Narrow_Eyes', 'Eyeglasses', 'Smiling', 'Mouth_Slightly_Open']

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=300000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='/media/limy/新加卷1/Datasets/Faces/CelebA/Celeba-HQ/celeba-128')
    parser.add_argument('--attr_path', type=str, default='/media/limy/新加卷1/Datasets/Faces/CelebA/Anno/list_attr_celebahq.txt')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
