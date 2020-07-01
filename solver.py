#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from model import Transform_Generator
from model import Transform_Discriminator
from model import Enhance_Generator
from model import Enhance_DiscriminatorC, Enhance_DiscriminatorT
from model import TVLoss
from model import GaussianBlur
from model import GrayLayer
from model import VGG
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import time
import datetime



class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader

        # Model configurations.
        self.c_dim = config.c_dim

        self.image_size = config.image_size
        self.magnification = config.magnification
        ## Transform Network configurations.
        self.tg_conv_dim = config.tg_conv_dim
        self.td_conv_dim = config.td_conv_dim
        self.tg_repeat_num = config.tg_repeat_num
        self.td_repeat_num = config.td_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        ## Transform Network configurations.
        
        
        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.tg_lr = config.tg_lr
        self.td_lr = config.td_lr
        self.eg_lr = config.eg_lr
        self.edc_lr = config.edc_lr
        self.edt_lr = config.edt_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # TV loss
        self.tv_criterion = TVLoss(config.tv_weight)
        # Color loss
        self.color_criterion = nn.CrossEntropyLoss()
        # Texture loss
        self.texture_criterion = nn.CrossEntropyLoss()
        # identity loss
        self.identity_criterion = nn.L1Loss()
        # content loss
        self.content_criterion = nn.MSELoss()
        # reconstruction loss
        self.rec_criterion = nn.MSELoss()

        # Enhancement Operations
        self.blur = GaussianBlur()
        self.gray = GrayLayer()
        self.vgg = VGG()

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create generators and discriminators: Transform_Generator(T_G), Transform_Discriminator(T_D), 
        Enhance_Generator(E_G), Enhance_Discriminator_color(E_Dc), Enhance_Discriminator_texture(E_Dt)."""

        self.T_G = Transform_Generator(self.tg_conv_dim, self.c_dim, self.tg_repeat_num)
        self.T_D = Transform_Discriminator(self.image_size, self.td_conv_dim, self.c_dim, self.td_repeat_num) 
        self.E_G = Enhance_Generator()
        
        self.tg_optimizer = torch.optim.Adam(self.T_G.parameters(), self.tg_lr, [self.beta1, self.beta2])
        self.td_optimizer = torch.optim.Adam(self.T_D.parameters(), self.td_lr, [self.beta1, self.beta2])
        self.eg_optimizer = torch.optim.Adam(self.E_G.parameters(), self.eg_lr, [self.beta1, self.beta2])
        
        self.print_network(self.T_G, 'T_G')
        self.print_network(self.T_D, 'T_D')
        self.print_network(self.E_G, 'E_G')
        
        self.T_G.to(self.device)
        self.T_D.to(self.device)
        self.E_G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        T_G_path = os.path.join(self.model_save_dir, '{}-T_G.ckpt'.format(resume_iters))
        T_D_path = os.path.join(self.model_save_dir, '{}-T_D.ckpt'.format(resume_iters))
        E_G_path = os.path.join(self.model_save_dir, '{}-E_G.ckpt'.format(resume_iters))

        self.T_G.load_state_dict(torch.load(T_G_path, map_location=lambda storage, loc: storage))
        self.T_D.load_state_dict(torch.load(T_D_path, map_location=lambda storage, loc: storage))
        self.E_G.load_state_dict(torch.load(E_G_path, map_location=lambda storage, loc: storage))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, tg_lr, td_lr, eg_lr, edc_lr, edt_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.tg_optimizer.param_groups:
            param_group['lr'] = tg_lr
        for param_group in self.td_optimizer.param_groups:
            param_group['lr'] = td_lr
        for param_group in self.eg_optimizer.param_groups:
            param_group['lr'] = eg_lr


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.tg_optimizer.zero_grad()
        self.td_optimizer.zero_grad()
        self.eg_optimizer.zero_grad()


    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.celeba_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_H_fixed, x_Lm_fixed, x_L_fixed, y_org = next(data_iter)
        x_H_fixed = x_H_fixed.to(self.device)
        x_Lm_fixed = x_Lm_fixed.to(self.device)
        x_L_fixed = x_L_fixed.to(self.device)
        y_trg_list = self.create_labels(y_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        tg_lr = self.tg_lr
        td_lr = self.td_lr
        eg_lr = self.eg_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_H, x_Lm, x_L, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_H, x_Lm, x_L, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            # lab

            y_org = label_org.clone()
            y_trg = label_trg.clone()

            x_H = x_H.to(self.device)                 # HR images
            x_Lm = x_Lm.to(self.device)               # LR images
            x_L = x_L.to(self.device)                 # Input images.
            y_org = y_org.to(self.device)             # Original domain labels.
            y_trg = y_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            
            # =================================================================================== #
            #                             2. Generators and Discriminators                        #
            # =================================================================================== #
            """
            # Transform_Generator
            x_T = self.T_G(x_L, y_trg)
            x_T_TG = self.T_G(x_T, y_org)
            
            # Transform_Discriminator & Classifier
            x_T_TD, x_T_TC = self.T_D(x_T)
            x_H_TD, x_H_TC = self.T_D(x_H)
            
            # Enhancement_Generator
            x_E = self.E_G(x_T)
            
            # Enhancement_VGG for identity loss
            x_E_vgg = self.vgg(x_E)
            x_H_vgg = self.vgg(x_H)
            
            # Enhancement_Discriminator_Color for color loss
            x_E_blur = self.blur(x_E)
            x_H_blur = self.blur(x_H)
            x_E_blur_EDc = self.E_Dc(x_E_blur)
            x_H_blur_EDc = self.E_Dc(x_H_blur)

            # Enhancement_Discriminator_Texture for texture loss
            x_E_gray = self.gray(x_E)
            x_H_gray = self.gray(x_H)
            x_E_gray_EDt = self.E_Dt(x_E_gray)
            x_H_gray_EDt = self.E_Dt(x_H_gray)
            """

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            
            # Discriminators
            ## Transform Network
            x_H_TD, x_H_TC = self.T_D(x_H)
            x_T = self.T_G(x_L, y_trg)
            x_T_TD, x_T_TC = self.T_D(x_T.detach())
            ### adv_loss
            td_loss_real = - torch.mean(x_H_TD)
            td_loss_fake = torch.mean(x_T_TD)
            td_loss_adv = td_loss_real + td_loss_fake
            ### att_loss
            td_loss_att = self.classification_loss(x_H_TC, label_org)
            ### gp_loss
            alpha = torch.rand(x_H.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_H.data + (1 - alpha) * x_T.data).requires_grad_(True)
            x_hat_TD, _ = self.T_D(x_hat)
            td_loss_gp = self.gradient_penalty(x_hat_TD, x_hat)
            ### T_D Total Loss
            td_loss = td_loss_adv + td_loss_att + td_loss_gp * 10
            ## Enhancement Network
            x_E = self.E_G(x_T)

            # Discriminators Loss
            d_loss = td_loss
            
            self.reset_grad()
            d_loss.backward()
            self.td_optimizer.step()

            
            # Logging.
            loss = {}
            loss['T_D/loss_TD'] = td_loss.item()
            loss['T_D/loss_adv'] = td_loss_adv.item()
            loss['T_D/loss_att'] = td_loss_att.item()
            loss['T_D/loss_gp'] = td_loss_gp.item()

            
            # =================================================================================== #
            #                               3. Train the generators                               #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                
                # Generators
                ## Transform Network
                x_T = self.T_G(x_L, y_trg)
                x_T_TD, x_T_TC = self.T_D(x_T)
                x_T_TG = self.T_G(x_T, y_org)
                ### adv_loss
                tg_loss_fake = - torch.mean(x_T_TD)
                tg_loss_adv = tg_loss_fake
                ### att_loss
                tg_loss_att = self.classification_loss(x_T_TC, label_trg)
                ### rec_loss
                # tg_loss_rec = torch.mean(torch.abs(x_H - x_T_TG))
                tg_loss_rec = self.rec_criterion(x_T_TG, x_H)

                ### T_G Total Loss
                tg_loss = tg_loss_adv + tg_loss_att + tg_loss_rec * 10 + tg_loss_tv

                ### identity_loss
                _, c1, h1, w1 = x_T.size()
                chw1 = c1 * h1 * w1
                eg_loss_identity = 1.0/chw1 * self.content_criterion(x_E_vgg, x_H_vgg)

                ### content_loss
                eg_loss_content = self.content_criterion(x_E, x_H)
                ### E_G Total Loss
                eg_loss = eg_loss_identity + eg_loss_content
                # Generators Loss
                g_loss = tg_loss + eg_loss
                
                self.reset_grad()
                g_loss.backward()
                self.tg_optimizer.step()
                self.eg_optimizer.step()
                
                # Logging.
                loss = {}
                loss['T_G/loss_TG'] = tg_loss.item()
                loss['T_G/loss_adv'] = tg_loss_adv.item()
                loss['T_G/loss_att'] = tg_loss_att.item()
                loss['T_G/loss_rec'] = tg_loss_rec.item()
                loss['E_G/loss_EG'] = eg_loss.item()
                loss['E_G/loss_identity'] = eg_loss_identity.item()
                loss['E_G/loss_content'] = eg_loss_content.item()
            
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_output_list = [x_H_fixed, x_Lm_fixed, x_L_fixed]
                    x_output_list.append(self.T_G(x_L_fixed, y_org))
                    x_output_list.append(self.E_G(self.T_G(x_L_fixed, y_org)))
                    for y_trg in y_trg_list:
                        x_output_list.append(self.T_G(x_L_fixed, y_trg))
                        x_output_list.append(self.E_G(self.T_G(x_L_fixed, y_trg)))
                    x_concat = torch.cat(x_output_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                T_G_path = os.path.join(self.model_save_dir, '{}-T_G.ckpt'.format(i+1))
                T_D_path = os.path.join(self.model_save_dir, '{}-T_D.ckpt'.format(i+1))
                E_G_path = os.path.join(self.model_save_dir, '{}-E_G.ckpt'.format(i+1))
                torch.save(self.T_G.state_dict(), T_G_path)
                torch.save(self.T_D.state_dict(), T_D_path)
                torch.save(self.E_G.state_dict(), E_G_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                tg_lr -= (self.tg_lr / float(self.num_iters_decay))
                td_lr -= (self.td_lr / float(self.num_iters_decay))
                eg_lr -= (self.eg_lr / float(self.num_iters_decay))
                self.update_lr(tg_lr, td_lr, eg_lr)
                print ('Decayed learning rates, tg_lr: {}, td_lr: {}, eg_lr: {}, edc_lr: {}, edt_lr: {}.'.format(tg_lr, td_lr, eg_lr, edc_lr, edt_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.celeba_loader

        with torch.no_grad():
            for i, (x_H, x_Lm, x_L, y_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_H = x_H.to(self.device)
                x_Lm = x_Lm.to(self.device)
                x_L = x_L.to(self.device)
                y_org = y_org.to(self.device)
                y_trg_list = self.create_labels(y_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_output_list = [x_H, x_Lm, x_L]
                x_output_list.append(self.T_G(x_L, y_org))
                x_output_list.append(self.E_G(self.T_G(x_L, y_org)))
                for y_trg in y_trg_list:
                    x_output_list.append(self.T_G(x_L, y_trg))
                    x_output_list.append(self.E_G(self.T_G(x_L, y_trg)))

                # Save the translated images.
                x_concat = torch.cat(x_output_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

