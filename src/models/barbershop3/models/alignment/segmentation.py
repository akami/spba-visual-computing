import torch
from torch import nn
import torchvision
from PIL import Image
import os
import numpy as np

from models.barbershop3.models.alignment.face_parsing.model import BiSeNet, seg_mean, seg_std
from models.barbershop3.utils.bicubic_downsampler import BicubicDownSampler
import models.barbershop3.utils.image_utils as utils


class Segmentation(nn.Module):
    """
    Initial semantic face segmentation using a modified BiSeNet network called Face Parsing.
    BiSeNet is a deep convolutional neural network based on ResNet,
    that is used for real-time semantic segmentation tasks.

    This section is adapted from Barbershop's alignment step. I decided to split the alignment phase into two:
    Segmentation and Mask Alignment.

    Let M_k = SEGMENT(I_k), where M_1 = face_identity_mask, M_2 = hair_identity_mask

    Inherits from Pytorch Neural Network Module base class to be included in the Neural Network.

    references:
    ** BiSeNet ** : https://arxiv.org/abs/1808.00897
    ** ResNet ** : https://arxiv.org/pdf/1512.03385.pdf
    ** FaceParsing ** : https://github.com/zllrunning/face-parsing.PyTorch
    """

    def __init__(self, model_path, tmp_dir, device='cuda', img_size=1024):
        super(Segmentation, self).__init__()

        # TODO parser opts?
        self.device = device
        self.img_size = img_size
        self.tmp_dir = tmp_dir

        self._load_segmentation_network(model_path)

        self.bicubic_downsample_512 = self._load_downsampling(512)
        self.bicubic_downsample_215 = self._load_downsampling(215)

    def _get_img_suffix(self):
        return "segmented"

    def run(self, *img_paths):

        # STEP 1: get face input segmentation mask

        face_img = self._preprocess_img(img_paths[0])
        face_segmentation = self._create_segmentation_mask(face_img)

        ## save intermediate results
        face_img_name, face_img_ext = os.path.splitext(os.path.basename(img_paths[0]))
        face_segmentation = face_segmentation[0].byte().cpu().detach()
        utils.save_seg_mask_to_img(face_img_name, self.tmp_dir, face_segmentation.squeeze().cpu())

        # STEP 2: get hair segmentation mask

        hair_img = self._preprocess_img(img_paths[1])
        hair_segmentation = self._create_segmentation_mask(hair_img)

        ## save intermediate results
        hair_img_name, hair_img_ext = os.path.splitext(os.path.basename(img_paths[1]))
        hair_segmentation = hair_segmentation[0].byte().cpu().detach()
        utils.save_seg_mask_to_img(hair_img_name, self.tmp_dir, hair_segmentation.squeeze().cpu())

        # TODO what does this do?
        OB_region = torch.where(
            (hair_segmentation != 10) * (hair_segmentation != 0) * (hair_segmentation != 15) * (
                    face_segmentation == 0),
            255 * torch.ones_like(face_segmentation), torch.zeros_like(face_segmentation))

    def _load_segmentation_network(self, model_path):
        # set up BiSeNet segmentation network
        self.segmentation_net = BiSeNet(n_classes=16)

        # run using specified device
        self.segmentation_net.to(self.device)

        # load weights from downloaded pre-trained model

        self.segmentation_net.load_state_dict(torch.load(model_path))

        for param in self.segmentation_net.parameters():
            param.requires_grad = False

        self.segmentation_net.eval()

    def _load_downsampling(self, size):
        """
        Setup of downsample methods
        :return: void
        """
        return BicubicDownSampler(factor=self.img_size // size)

    def _preprocess_img(self, img_path):
        """
        Preprocess image to be fed into segmentation network.
        Steps include:
            1. open as PIL Image
            2. transform PIL Image to Pytorch Tensor data type
            3. downsample Tensor to 512x512
        :param img_path:
        :return: 4D tensor of shape (1, 3, 512, 512)
        """
        # open as PIL Image
        img = Image.open(img_path)

        # transform PIL Image to Pytorch Tensor, unsqueeze(add dimension), to type cuda/cpu
        img_tensor = torchvision.transforms.ToTensor()(img)[:3].unsqueeze(0).to(self.device)

        # downsample img tensor to 512x512
        img_tensor = (self.bicubic_downsample_512(img_tensor).clamp(0, 1)) - seg_mean / seg_std

        return img_tensor

    def _create_segmentation_mask(self, img):
        """
        Creates segmentation mask

        semantic regions in the first row of the tensor:
        (0: background, 1: skin, 2: nose, 3: ???, 4: eyes, 5: eyebrows,
         6: ears, 7: teeth, 8: upper lip, 9: lower lip, 10: hair,
         11: ???, 12: earrings, 13: ???, 14: neck, 15: clothing)
        :param img: bicubically downsampled img tensor of shape (1, 3, 512, 512)
        :return: 3D tensor of shape (1, 512, 512)
        """
        down_seg1, _, _ = self.segmentation_net(img)   # returns tensor of shape (1, 16, 512, 512) --> 16 semantic regions
        seg_target1 = torch.argmax(down_seg1, dim=1).long() # reduce dimension to tensor of shape (1, 512, 512)

        return seg_target1
