"""
RL-Deconv module for test
Created on Tue Sep 30 19:31:36 2023
@ Last Updated by Lin Yuhuan
"""
import torch
import itertools
from .base_model import BaseModel
from . import networks
from .section import *
from .read_mrc import write_mrc
from .Unet_G import Unet_deconv
from .V_Net import Feature_Net
from .Patch_GAN_D import NLayerDiscriminator
from math import pi
from .RL_Deconv import RL_Deconv
from .FP_Net import RNL
from .TV_loss import TV_Loss


class DualGANASectionModel(BaseModel):
    """ RL-DFN Model, add function update_crop_size for test
        Args:
            patch_x (int),patch_y (int), patch_z (int): patch size .
            gpu_ids (list(int)): GPU_device number
            gan_mode (str): choose the GAN loss type.
            name (str): Experiment Name.
            lambda_plane (list(int)): weight of lambda_plane_target, lambda_slice, lambda_proj
            output_nc (int), input_nc (input): output/input channel size
            psfA_root(str)，psfB_root(str): two views' estimated PSF root, TIFF file
            lr (float), beta1(float) : optimizer parameters
            batch_size (int): batch_size
            TV_loss_weight (int): weight of TV Loss
            cycle_loss_weight (int): The indentix of Block in each RG
            patch_x_re (int),patch_y_re (int): the residue part of BIG Image
    """
    def __init__(self, patch_x=128, patch_y=128, patch_z=64, gpu_ids=[3], gan_mode='vanilla', name="experiment name",
                 randomize_projection_depth='store_true',
                 lambda_plane=[1, 1, 1], output_nc=1, input_nc=1, device="cuda:3", psfA_root="", psfB_root="",
                 lr=0.0001, beta1=0.1, batch_size=1, TV_loss_weight=0.001, cycle_loss_weight=50,patch_x_re = None,patch_y_re = None):
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_z = patch_z
        self.TV_loss_weight = TV_loss_weight
        self.cycle_loss_weight = cycle_loss_weight

        BaseModel.__init__(self, name=name, gpu_ids=gpu_ids)
        self.loss_names = ['D_A', 'G_A', 'G_B', 'cycle1', 'cycle2', 'D_B']
        self.gan_mode = gan_mode
        self.Two_discrimision = False

        self.view = 2
        self.rot_fake = False
        self.lambda_G = 10
        self.unrot = False
        self.total_iters = 0
        self.save_freq = 20
        self.isTrain = True
        self.device = device
        self.batch_size = batch_size

        self.gen_dimension = 3  # 3D convolutions in generators
        self.dis_dimension = 2  # 2D convolutions in discriminators

        self.randomize_projection_depth = randomize_projection_depth

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real', 'fake', 'rec']
        visual_names_B = ['real', 'fake', 'rec']

        self.lambda_plane_target, self.lambda_slice, self.lambda_proj = [
            factor / (lambda_plane[0] + lambda_plane[1] + lambda_plane[2]) for factor in lambda_plane]

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        self.lateral_axis = 0  # XY plane
        self.axial_1_axis = 1  # XZ plane
        self.axial_2_axis = 2  # YZ plane

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'D_A', 'D_B', 'F_A', "F_B"]
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        self.input_nc_g = input_nc
        self.output_nc_g = output_nc

        norm_layer = networks.get_norm_layer(norm_type="batch", dimension=3)
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_list = []
        self.netF_A_list = []
        self.netF_B_list = []
        self.RNL_Net_list = []
        deconv_iter = 1
        self.deconv_iter = deconv_iter

        for i in range(deconv_iter):
            self.netF_A_list.append(
                Feature_Net(input_nc=self.input_nc_g, output_nc=64, norm_layer=norm_layer, dimension=3).to(self.device))
            self.netF_B_list.append(
                Feature_Net(input_nc=self.input_nc_g, output_nc=64, norm_layer=norm_layer, dimension=3).to(self.device))
            if i > 0:
                self.RNL_Net_list.append(RNL(norm_layer=norm_layer).to(self.device))

            self.netG_list.append(
                Unet_deconv(input_nc=64, output_nc=self.output_nc_g, norm_layer=norm_layer, dimension=3,
                            feature_input=True).to(device))
        self.RL_deconv_list = [0,0,0,0]
        """
        use different crop size
        """
        self.RL_deconv_list0=(RL_Deconv(psfA_root=psfA_root, psfB_root=psfB_root, X_dim=patch_x, Y_dim=patch_y,
                                   Z_dim=patch_z,
                                   device=device, batch_size=batch_size))
        self.RL_deconv_list1=(RL_Deconv(psfA_root=psfA_root, psfB_root=psfB_root, X_dim=patch_x_re, Y_dim=patch_y,
                                             Z_dim=patch_z,
                                             device=device, batch_size=batch_size))
        self.RL_deconv_list2=(RL_Deconv(psfA_root=psfA_root, psfB_root=psfB_root, X_dim=patch_x, Y_dim=patch_y_re,
                                             Z_dim=patch_z,
                                             device=device, batch_size=batch_size))
        self.RL_deconv_list3=(RL_Deconv(psfA_root=psfA_root, psfB_root=psfB_root, X_dim=patch_x_re, Y_dim=patch_y_re,
                                             Z_dim=patch_z,
                                             device=device, batch_size=batch_size))

        self.if_x = 0
        self.if_y = 0

        self.RL_deconv_list0.set_device(device)
        self.RL_deconv_list1.set_device(device)
        self.RL_deconv_list2.set_device(device)
        self.RL_deconv_list3.set_device(device)

        norm_layer = networks.get_norm_layer(norm_type="batch", dimension=2)

        self.netD_A = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer, use_sigmoid=True,
                                          dimension=2, use_bias=True).to(device)
        self.netD_B = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=norm_layer, use_sigmoid=True,
                                          dimension=2, use_bias=True).to(device)

        self.criterionGAN = networks.GANLoss(gan_mode).to(self.device)  # define GAN loss.
        self.criterionCycleA = torch.nn.L1Loss()
        self.criterionCycleB = torch.nn.L1Loss()
        self.TV_loss = TV_Loss()

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        if deconv_iter == 1:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_list[0].parameters(), self.netF_A_list[0].parameters(),
                                self.netF_B_list[0].parameters()), lr=lr, betas=(beta1, 0.999))
        elif deconv_iter == 2:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_list[0].parameters(), self.netF_A_list[0].parameters(),
                                self.netF_B_list[0].parameters(), self.netG_list[1].parameters(),
                                self.netF_A_list[1].parameters(),
                                self.netF_B_list[1].parameters(), self.RNL_Net_list[0].parameters()), lr=lr,
                betas=(beta1, 0.999))
        else:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_list[0].parameters(), self.netF_A_list[0].parameters(),
                                self.netF_B_list[0].parameters(), self.netG_list[1].parameters(),
                                self.netF_A_list[1].parameters(),
                                self.netF_B_list[1].parameters(), self.netG_list[2].parameters(),
                                self.netF_A_list[2].parameters(),
                                self.netF_B_list[2].parameters(), self.RNL_Net_list[0].parameters(),
                                self.RNL_Net_list[1].parameters()), lr=lr,
                betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=0.0001, betas=(beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        ## END OF INITIALIZATION ##
    def section_prepare(self, H=None, W=None, head=None, pixel_off=0, theta_off=0, theta=30.8):

        self.head = head

        H1, W1, dire1, off1 = get_adptive_size(X_dim=self.patch_x, Y_dim=self.patch_y, Z_dim=self.patch_z, axis="Y",
                                               theta=pi * (theta / 180))
        H2, W2, dire2, off2 = get_adptive_size(X_dim=self.patch_x, Y_dim=self.patch_y, Z_dim=self.patch_z, axis="Y",
                                               theta=pi * ((180 - theta) / 180))

        if H == None:
            H = min(H1, H2)
        if W == None:
            W = min(W1, W2)

        self.head[0][0] = H
        self.head[0][1] = W
        self.head[0][3] = 2

        self.H = H
        self.W = W
        cor = section_position(H, W, axis="Y", theta=pi * (theta / 180)) + np.array([pixel_off, 0, 0]).reshape(1, -1)
        self.d_cor_Y_pi_6 = discret_cor(cor).astype(int)
        self.bucubic_Y_pi_6_numpy = Bicubic_function(cor, method="linear").astype(np.float32)
        self.bucubic_Y_pi_6 = torch.from_numpy(self.bucubic_Y_pi_6_numpy).to(self.device)
        self.off_Y_pi_6 = floor(self.patch_x - H * cos(pi * (theta / 180))) - 10

        cor = section_position(H, W, axis="Y", theta=pi * ((180 - theta) / 180)) + np.array([pixel_off, 0, 0]).reshape(
            1,
            -1)
        self.bucubic_Y_5pi_6_numpy = Bicubic_function(cor, method="linear").astype(np.float32)
        self.bucubic_Y_5pi_6 = torch.from_numpy(self.bucubic_Y_5pi_6_numpy).to(self.device)
        self.d_cor_Y_5pi_6 = discret_cor(cor).astype(int)
        self.off_Y_5pi_6 = floor(self.patch_x - H * cos(pi * (180 - theta / 180))) - 10

        cor = section_position(H, W, axis="Y", theta=pi * (theta / 180) + theta_off) + np.array(
            [pixel_off, 0, 0]).reshape(1, -1)
        self.d_cor_A = discret_cor(cor).astype(int)
        self.bucubic_A_numpy = Bicubic_function(cor, method="linear").astype(np.float32)
        self.bucubic_A = torch.from_numpy(self.bucubic_A_numpy).to(self.device)
        self.off_A = max(floor(self.patch_x - H * abs(cos(pi * (theta / 180) + theta_off))) - 10, 0)

        cor = section_position(H, W, axis="Y", theta=pi * ((180 - theta) / 180) + theta_off) + np.array(
            [pixel_off, 0, 0]).reshape(1,
                                       -1)
        self.bucubic_B_numpy = Bicubic_function(cor, method="linear").astype(np.float32)
        self.bucubic_B = torch.from_numpy(self.bucubic_B_numpy).to(self.device)
        self.d_cor_B = discret_cor(cor).astype(int)
        self.off_B = max(floor(self.patch_x - H * abs(cos(pi * ((180 - theta) / 180) + theta_off))) - 10, 0)

        cor = section_position(H, W, axis="Y", theta=pi * (0.5)) + np.array(
            [pixel_off, 0, 0]).reshape(1,
                                       -1)
        self.bucubic_C_numpy = Bicubic_function(cor, method="linear").astype(np.float32)
        self.bucubic_C = torch.from_numpy(self.bucubic_C_numpy).to(self.device)
        self.d_cor_C = discret_cor(cor).astype(int)
        self.off_C = max(floor(self.patch_x - 10), 0)

        self.section_dict = {}


    def update_crop_size(self,patch_x,patch_y,patch_z,if_x,if_y):
        """
        whole cell may be cropped in different size, need change crop size in model
        """
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_z = patch_z
        self.if_x=if_x
        self.if_y = if_y


    def set_eval(self):
        self.netF_A_list[0].eval()
        self.netF_B_list[0].eval()
        self.netG_list[0].eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """

        self.real_rot1 = input[0].to(self.device)
        self.real_rot2 = input[1].to(self.device)
        self.fusion = input[2].to(self.device)
        self.real = torch.cat((self.real_rot1, self.real_rot2), dim=1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>.
        In this version, we iterate through each slice in a cube.
        """

        real_rot1 = self.real_rot1.reshape(self.batch_size, self.patch_z, self.patch_y, self.patch_x)
        real_rot2 = self.real_rot2.reshape(self.batch_size, self.patch_z, self.patch_y, self.patch_x)
        fusion = (real_rot1 + real_rot2) / 2
        for i in range(self.deconv_iter):
            if self.if_x == 0 and self.if_y == 0:
                cur1, cur2 = self.RL_deconv_list0(real_rot1, real_rot2, fusion)

            elif self.if_x != 0 and self.if_y == 0:
                cur1, cur2= self.RL_deconv_list1(real_rot1, real_rot2, fusion)
            elif self.if_x == 0 and self.if_y != 0:
                cur1, cur2 = self.RL_deconv_list2(real_rot1, real_rot2, fusion)
            else:
                cur1, cur2 = self.RL_deconv_list3(real_rot1, real_rot2, fusion)

            cur1 = cur1.reshape(self.batch_size, 1, self.patch_z, self.patch_y, self.patch_x).detach()
            cur2 = cur2.reshape(self.batch_size, 1, self.patch_z, self.patch_y, self.patch_x).detach()
            v1 = self.netF_A_list[i](cur1)
            v2 = self.netF_A_list[i](cur2)
            fusion = v1 * v2
            fusion = self.netG_list[i](fusion)  # G_A(A)
        self.fake = fusion
        self.degeration_A, self.degeration_B = self.RL_deconv_list0.degradation(self.fake)

    def visualization(self, root):
        real_root1 = root + "/total_iets" + str(self.total_iters) + "real1.mrc"
        real_root2 = root + "/total_iets" + str(self.total_iters) + "real2.mrc"
        fake_root = root + "/total_iets" + str(self.total_iters) + "fake.mrc"
        fusion_root = root + "/total_iets" + str(self.total_iters) + "fusion.mrc"
        degeration_A_root = root + "/total_iets" + str(self.total_iters) + "degeration_A.mrc"
        degeration_B_root = root + "/total_iets" + str(self.total_iters) + "degeration_B.mrc"

        if self.total_iters % self.save_freq == 0:
            self.head[0][0] = self.patch_x
            self.head[0][1] = self.patch_y
            self.head[0][2] = self.patch_z
            self.head[0][3] = 2
            real1 = self.real_rot1[0, :, :, :, :].squeeze(dim=0).squeeze(dim=0).permute(2, 1, 0)
            real1 = real1.cpu().detach().numpy()
            real2 = self.real_rot2[0, :, :, :, :].squeeze(dim=0).squeeze(dim=0).permute(2, 1, 0)
            real2 = real2.cpu().detach().numpy()
            fake = self.fake[0, :, :, :, :].squeeze(dim=0).squeeze(dim=0).permute(2, 1, 0)
            fake = fake.cpu().detach().numpy()
            fusion = self.fusion[0, :, :, :, :].squeeze(dim=0).squeeze(dim=0).permute(2, 1, 0)
            fusion = fusion.cpu().detach().numpy()

            degeration_A = self.degeration_A[0, :, :, :, :].squeeze(dim=0).squeeze(dim=0).permute(2, 1, 0)
            degeration_A = degeration_A.cpu().detach().numpy()
            degeration_B = self.degeration_B[0, :, :, :, :].squeeze(dim=0).squeeze(dim=0).permute(2, 1, 0)
            degeration_B = degeration_B.cpu().detach().numpy()

            write_mrc(real_root1, real1, self.head)
            write_mrc(real_root2, real2, self.head)
            write_mrc(fake_root, fake, self.head)
            write_mrc(fusion_root, fusion, self.head)
            write_mrc(degeration_A_root, degeration_A, self.head)
            write_mrc(degeration_B_root, degeration_B, self.head)

    def backward_D_slice(self, netD, real, fake, slice_axis_real, slice_axis_fake):

        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real

        pred_real = self.iter_f_section(real, netD, slice_axis_real, real=True)

        pred_fake = self.iter_f_section(fake.detach(), netD, slice_axis_fake, real=True)

        # real
        loss_D_real = self.criterionGAN(pred_real, True)  # Target_is_real -> True: loss (pred_real - unit vector)

        # Fake
        loss_D_fake = self.criterionGAN(pred_fake, False)  # no loss with the unit vector

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_slice(self.netD_A, self.real_rot1, self.fake, 0, 2) * 0.5

    #  self.loss_D_A += self.backward_D_slice(self.netD_A, self.real_rot1, self.fake, 0, 5) * 0.5

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_slice(self.netD_B, self.real_rot2, self.fake, 1, 3) * 0.5

    # self.loss_D_B += self.backward_D_slice(self.netD_B, self.real_rot2, self.fake, 1, 5) * 0.5

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_G = self.lambda_G

        self.loss_G_A = self.criterionGAN(self.iter_f_section(self.fake, self.netD_A, 2),
                                          True) * self.lambda_plane_target

        self.loss_G_B = self.criterionGAN(self.iter_f_section(self.fake, self.netD_B, 3),
                                          True) * self.lambda_plane_target

        self.loss_G_A += self.criterionGAN(self.iter_f_section(self.fake, self.netD_A, 4),
                                           True) * self.lambda_plane_target

        self.loss_G_B += self.criterionGAN(self.iter_f_section(self.fake, self.netD_B, 4),
                                           True) * self.lambda_plane_target

        # This model only includes forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle1 = self.criterionCycleA(self.degeration_A, self.real_rot1) * lambda_G
        self.loss_cycle2 = self.criterionCycleB(self.degeration_B, self.real_rot2) * lambda_G

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + (
                    self.loss_cycle1 + self.loss_cycle2) * self.cycle_loss_weight + self.TV_loss_weight * self.TV_loss(
            self.fake)

        # self.loss_G = (self.loss_cycle1 + self.loss_cycle2) * 50
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad(
            [self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()
        self.backward_D_B()  # calculate gradients for D_A's
        self.optimizer_D.step()  # update D_A and D_B's weights


    def iter_f_section(self, input, function, slice_axis, real=False):
        if slice_axis != 6:
            input = input.reshape(self.batch_size, self.patch_z, self.patch_y, self.patch_x).permute(0, 3, 2, 1)
            if slice_axis == 1:
                d_cor = None
                for i in range(self.batch_size):
                    off1 = random.randint(0, self.off_Y_pi_6)
                    off2 = random.randint(0, 1)
                    d_cor1 = self.d_cor_Y_pi_6 + np.array([off1, 0, off2]).reshape(1, 1, 1, 1, 1, 3)
                    if i == 0:
                        d_cor = d_cor1
                    else:
                        d_cor = np.concatenate((d_cor, d_cor1), axis=0)

                d_cor = d_cor.astype(int)

                batch_index_ = np.ones_like(d_cor1[:, :, :, :, :, 0])
                batch_index = None
                for i in range(self.batch_size):
                    if i == 0:
                        batch_index = i * batch_index_
                    else:
                        batch_index = np.concatenate((batch_index, i * batch_index_), axis=0)

                # section = get_section(input,d_cor.astype(int),self.H,self.W,self.bucubic_Y_5pi_6,batch=1).astype(np.float32).reshape(-1,1,self.H,self.W).transpose(0,1,3,2)
                section_near = input[
                    batch_index, d_cor[:, :, :, :, :, 0], d_cor[:, :, :, :, :, 1], d_cor[:, :, :, :, :, 2]]
                w_section_near = section_near * (self.bucubic_Y_pi_6).repeat(self.batch_size, 1, 1, 1, 1)
                w_section_near = w_section_near.reshape(self.batch_size, -1, 4 * 4 * 4)
                w_section_near = w_section_near.sum(dim=2)
                section = w_section_near.reshape(self.batch_size, self.H, self.W).reshape(-1, 1, self.H,
                                                                                          self.W).permute(0, 1, 3, 2)

            elif slice_axis == 0:
                d_cor = None
                for i in range(self.batch_size):
                    off1 = random.randint(0, self.off_Y_5pi_6)
                    off2 = random.randint(0, 1)
                    d_cor1 = self.d_cor_Y_5pi_6 + np.array([off1, 0, off2]).reshape(1, 1, 1, 1, 1, 3)
                    if i == 0:
                        d_cor = d_cor1
                    else:
                        d_cor = np.concatenate((d_cor, d_cor1), axis=0)

                d_cor = d_cor.astype(int)
                batch_index_ = np.ones_like(d_cor1[:, :, :, :, :, 0])
                batch_index = None
                for i in range(self.batch_size):
                    if i == 0:
                        batch_index = i * batch_index_
                    else:
                        batch_index = np.concatenate((batch_index, i * batch_index_), axis=0)

                section_near = input[
                    batch_index, d_cor[:, :, :, :, :, 0], d_cor[:, :, :, :, :, 1], d_cor[:, :, :, :, :, 2]]
                w_section_near = section_near * (self.bucubic_Y_5pi_6).repeat(self.batch_size, 1, 1, 1, 1)
                w_section_near = w_section_near.reshape(self.batch_size, -1, 4 * 4 * 4)
                w_section_near = w_section_near.sum(dim=2)
                section = w_section_near.reshape(self.batch_size, self.H, self.W).reshape(-1, 1, self.H,
                                                                                          self.W).permute(0, 1, 3, 2)

            if slice_axis == 2:
                d_cor = None
                for i in range(self.batch_size):
                    off1 = random.randint(0, self.off_A)
                    off2 = random.randint(0, 1)
                    d_cor1 = self.d_cor_A + np.array([off1, 0, off2]).reshape(1, 1, 1, 1, 1, 3)
                    if i == 0:
                        d_cor = d_cor1
                    else:
                        d_cor = np.concatenate((d_cor, d_cor1), axis=0)

                d_cor = d_cor.astype(int)
                batch_index_ = np.ones_like(d_cor1[:, :, :, :, :, 0])
                batch_index = None
                for i in range(self.batch_size):
                    if i == 0:
                        batch_index = i * batch_index_
                    else:
                        batch_index = np.concatenate((batch_index, i * batch_index_), axis=0)

                section_near = input[
                    batch_index, d_cor[:, :, :, :, :, 0], d_cor[:, :, :, :, :, 1], d_cor[:, :, :, :, :, 2]]
                w_section_near = section_near * (self.bucubic_A).repeat(self.batch_size, 1, 1, 1, 1)
                w_section_near = w_section_near.reshape(1, -1, 4 * 4 * 4)
                w_section_near = w_section_near.sum(dim=2)
                section = w_section_near.reshape(self.batch_size, self.H, self.W).reshape(-1, 1, self.H,
                                                                                          self.W).permute(0, 1, 3, 2)

            elif slice_axis == 3:
                d_cor = None
                for i in range(self.batch_size):
                    off1 = random.randint(0, self.off_B)
                    off2 = random.randint(0, 1)
                    d_cor1 = self.d_cor_B + np.array([off1, 0, off2]).reshape(1, 1, 1, 1, 1, 3)
                    if i == 0:
                        d_cor = d_cor1
                    else:
                        d_cor = np.concatenate((d_cor, d_cor1), axis=0)

                d_cor = d_cor.astype(int)
                batch_index_ = np.ones_like(d_cor1[:, :, :, :, :, 0])
                batch_index = None
                for i in range(self.batch_size):
                    if i == 0:
                        batch_index = i * batch_index_
                    else:
                        batch_index = np.concatenate((batch_index, i * batch_index_), axis=0)

                section_near = input[
                    batch_index, d_cor[:, :, :, :, :, 0], d_cor[:, :, :, :, :, 1], d_cor[:, :, :, :, :, 2]]
                w_section_near = section_near * (self.bucubic_B).repeat(self.batch_size, 1, 1, 1, 1)
                w_section_near = w_section_near.reshape(self.batch_size, -1, 4 * 4 * 4)
                w_section_near = w_section_near.sum(dim=2)
                section = w_section_near.reshape(self.batch_size, self.H, self.W).reshape(-1, 1, self.H,
                                                                                          self.W).permute(0, 1, 3, 2)

            elif slice_axis == 4:
                d_cor = None
                for i in range(self.batch_size):
                    off1 = random.randint(0, self.off_C)
                    off2 = random.randint(0, 1)
                    d_cor1 = self.d_cor_C + np.array([off1, 0, off2]).reshape(1, 1, 1, 1, 1, 3)
                    if i == 0:
                        d_cor = d_cor1
                    else:
                        d_cor = np.concatenate((d_cor, d_cor1), axis=0)

                d_cor = d_cor.astype(int)
                batch_index_ = np.ones_like(d_cor1[:, :, :, :, :, 0])
                batch_index = None
                for i in range(self.batch_size):
                    if i == 0:
                        batch_index = i * batch_index_
                    else:
                        batch_index = np.concatenate((batch_index, i * batch_index_), axis=0)
                section_near = input[
                    batch_index, d_cor[:, :, :, :, :, 0], d_cor[:, :, :, :, :, 1], d_cor[:, :, :, :, :, 2]]
                w_section_near = section_near * (self.bucubic_C).repeat(self.batch_size, 1, 1, 1, 1)
                w_section_near = w_section_near.reshape(1, -1, 4 * 4 * 4)
                w_section_near = w_section_near.sum(dim=2)
                section = w_section_near.reshape(self.batch_size, self.H, self.W).reshape(-1, 1, self.H,
                                                                                          self.W).permute(0, 1, 3, 2)

            elif slice_axis == 5:
                d_cor = None
                for i in range(self.batch_size):
                    off1 = random.randint(0, self.off_C)
                    off2 = random.randint(0, 1)
                    d_cor1 = self.d_cor_C + np.array([off1, 0, off2]).reshape(1, 1, 1, 1, 1, 3)
                    if i == 0:
                        d_cor = d_cor1
                    else:
                        d_cor = np.concatenate((d_cor, d_cor1), axis=0)

                d_cor = d_cor.astype(int)
                batch_index_ = np.ones_like(d_cor1[:, :, :, :, :, 0])
                batch_index = None
                for i in range(self.batch_size):
                    if i == 0:
                        batch_index = i * batch_index_
                    else:
                        batch_index = np.concatenate((batch_index, i * batch_index_), axis=0)
                input = input.permute(0, 2, 1, 3)
                section_near = input[
                    batch_index, d_cor[:, :, :, :, :, 0], d_cor[:, :, :, :, :, 1], d_cor[:, :, :, :, :, 2]]
                w_section_near = section_near * (self.bucubic_C).repeat(self.batch_size, 1, 1, 1, 1)
                w_section_near = w_section_near.reshape(1, -1, 4 * 4 * 4)
                w_section_near = w_section_near.sum(dim=2)
                section = w_section_near.reshape(self.batch_size, self.H, self.W).reshape(-1, 1, self.H,
                                                                                          self.W).permute(0, 1, 3, 2)

            output_slice = function(section)

            return output_slice

    def save_checkpoint(self, dataroot):
        if self.deconv_iter == 1:
            torch.save(self.netF_B_list[0].state_dict(), dataroot + 'net_FB.pth')
            torch.save(self.netF_A_list[0].state_dict(), dataroot + 'net_FA.pth')
            torch.save(self.netG_list[0].state_dict(), dataroot + 'net_G.pth')
            torch.save(self.netD_A.state_dict(), dataroot + 'netD_A.pth')
            torch.save(self.netD_B.state_dict(), dataroot + 'netD_B.pth')
        elif self.deconv_iter == 2:
            torch.save(self.netF_B_list[0].state_dict(), dataroot + 'net_FB.pth')
            torch.save(self.netF_A_list[0].state_dict(), dataroot + 'net_FA.pth')
            torch.save(self.netG_list[0].state_dict(), dataroot + 'net_G.pth')
            torch.save(self.netF_B_list[1].state_dict(), dataroot + 'net_FB1.pth')
            torch.save(self.netF_A_list[1].state_dict(), dataroot + 'net_FA1.pth')
            torch.save(self.netG_list[1].state_dict(), dataroot + 'net_G1.pth')
            torch.save(self.RNL_Net_list[0].state_dict(), dataroot + 'net_RNL.pth')
            torch.save(self.netD_A.state_dict(), dataroot + 'netD_A.pth')
            torch.save(self.netD_B.state_dict(), dataroot + 'netD_B.pth')
        else:
            torch.save(self.netF_B_list[0].state_dict(), dataroot + 'net_FB.pth')
            torch.save(self.netF_A_list[0].state_dict(), dataroot + 'net_FA.pth')
            torch.save(self.netG_list[0].state_dict(), dataroot + 'net_G.pth')
            torch.save(self.netF_B_list[1].state_dict(), dataroot + 'net_FB1.pth')
            torch.save(self.netF_A_list[1].state_dict(), dataroot + 'net_FA1.pth')
            torch.save(self.netG_list[1].state_dict(), dataroot + 'net_G1.pth')
            torch.save(self.RNL_Net_list[0].state_dict(), dataroot + 'net_RNL.pth')
            torch.save(self.netF_B_list[2].state_dict(), dataroot + 'net_FB1.pth')
            torch.save(self.netF_A_list[2].state_dict(), dataroot + 'net_FA1.pth')
            torch.save(self.netG_list[2].state_dict(), dataroot + 'net_G1.pth')
            torch.save(self.RNL_Net_list[1].state_dict(), dataroot + 'net_RNL.pth')
            torch.save(self.netD_A.state_dict(), dataroot + 'netD_A.pth')
            torch.save(self.netD_B.state_dict(), dataroot + 'netD_B.pth')

    def load_checkpoint(self, dataroot, map_location=None):
        if self.deconv_iter == 1:
            self.netF_A_list[0].load_state_dict(torch.load(dataroot + 'net_FA.pth', map_location=map_location,weights_only=False))
            self.netF_B_list[0].load_state_dict(torch.load(dataroot + 'net_FB.pth', map_location=map_location,weights_only=False))
            self.netG_list[0].load_state_dict(torch.load(dataroot + 'net_G.pth', map_location=map_location,weights_only=False))
            self.netD_A.load_state_dict(torch.load(dataroot + 'netD_A.pth', map_location=map_location,weights_only=False))
            self.netD_B.load_state_dict(torch.load(dataroot + 'netD_B.pth', map_location=map_location,weights_only=False))
        elif self.deconv_iter == 2:
            self.netF_A_list[0].load_state_dict(torch.load(dataroot + 'net_FA.pth', map_location=map_location))
            self.netF_B_list[0].load_state_dict(torch.load(dataroot + 'net_FB.pth', map_location=map_location))
            self.netG_list[0].load_state_dict(torch.load(dataroot + 'net_G.pth', map_location=map_location))
            self.netF_B_list[1].load_state_dict(torch.load(dataroot + 'net_FB1.pth', map_location=map_location))
            self.netF_A_list[1].load_state_dict(torch.load(dataroot + 'net_FA1.pth', map_location=map_location))
            self.netG_list[1].load_state_dict(torch.load(dataroot + 'net_G1.pth', map_location=map_location))
            self.RNL_Net_list[0].load_state_dict(torch.load(dataroot + 'net_RNL.pth', map_location=map_location))
            self.netD_A.load_state_dict(torch.load(dataroot + 'netD_A.pth', map_location=map_location))
            self.netD_B.load_state_dict(torch.load(dataroot + 'netD_B.pth', map_location=map_location))
        else:
            self.netF_A_list[0].load_state_dict(torch.load(dataroot + 'net_FA.pth', map_location=map_location))
            self.netF_B_list[0].load_state_dict(torch.load(dataroot + 'net_FB.pth', map_location=map_location))
            self.netG_list[0].load_state_dict(torch.load(dataroot + 'net_G.pth', map_location=map_location))
            self.netF_B_list[1].load_state_dict(torch.load(dataroot + 'net_FB1.pth', map_location=map_location))
            self.netF_A_list[1].load_state_dict(torch.load(dataroot + 'net_FA1.pth', map_location=map_location))
            self.netG_list[1].load_state_dict(torch.load(dataroot + 'net_G1.pth', map_location=map_location))
            self.RNL_Net_list[0].load_state_dict(torch.load(dataroot + 'net_RNL.pth', map_location=map_location))
            self.netF_B_list[2].load_state_dict(torch.load(dataroot + 'net_FB1.pth', map_location=map_location))
            self.netF_A_list[2].load_state_dict(torch.load(dataroot + 'net_FA1.pth', map_location=map_location))
            self.netG_list[2].load_state_dict(torch.load(dataroot + 'net_G1.pth', map_location=map_location))
            self.RNL_Net_list[1].load_state_dict(torch.load(dataroot + 'net_RNL.pth', map_location=map_location))
            self.netD_A.load_state_dict(torch.load(dataroot + 'netD_A.pth', map_location=map_location))
            self.netD_B.load_state_dict(torch.load(dataroot + 'netD_B.pth', map_location=map_location))


class Volume():
    def __init__(self, vol, device):
        self.volume = vol.to(device)  # push the volume to cuda memory
        self.num_slicex = vol.shape[-1]
        self.num_slicey = vol.shape[-2]
        self.num_slicez = vol.shape[-3]

    # returns a slice: # batch, color_channel, y, x
    def get_slice(self, slice_axis):
        slice_index_pickx = np.random.randint(self.num_slicex)
        slice_index_picky = np.random.randint(self.num_slicey)
        slice_index_pickz = np.random.randint(self.num_slicez)
        if slice_axis == 0:
            return self.volume[:, :, slice_index_pickz, :, :]

        elif slice_axis == 1:
            return self.volume[:, :, :, slice_index_picky, :]

        elif slice_axis == 2:
            return self.volume[:, :, :, :, slice_index_pickx]

    def get_projection(self, depth, slice_axis):
        start_index_x = np.random.randint(0, self.num_slicex - depth)
        start_index_y = np.random.randint(0, self.num_slicey - depth)
        start_index_z = np.random.randint(0, self.num_slicez - depth)
        if slice_axis == 0:
            volume_ROI = self.volume[:, :, start_index_z:start_index_z + depth, :, :]

        elif slice_axis == 1:
            volume_ROI = self.volume[:, :, :, start_index_y:start_index_y + depth, :]

        elif slice_axis == 2:
            volume_ROI = self.volume[:, :, :, :, start_index_x:start_index_x + depth]

        mip = torch.max(volume_ROI, dim=slice_axis + 2)[0]
        return mip

    def get_volume(self):
        return self.volume
