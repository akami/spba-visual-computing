import os
import torch
import torchvision.transforms as transforms
import PIL.Image
from PIL import ImageFile
import numpy as np
from argparse import Namespace
import cv2

from models.model import Model

from models.baldification.models.HairMapper.encoder4editing.models.psp import pSp
from models.baldification.models.HairMapper.styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from models.baldification.models.HairMapper.mapper.networks.level_mapper import LevelMapper
from models.baldification.models.HairMapper.classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet


class Baldification(Model):
    """
    This class represents the HairMapper model that is used as a preprocessing-step. It should be used to process the
    input face identity image in order to minimize the appearance of artifacts around the hair area.
    By eliminating the hair in the face identity image, we get a bald image.
    Note that, in order to get a realistic result, the image needs to be aligned according to the FFHQ dataset.
    """

    def __init__(self, loader, model_dir, tmp_dir):
        super().__init__(loader, model_dir, tmp_dir)

    def _get_img_suffix(self):
        return "bald"

    def _get_pretrained_models(self):
        return {
            'StyleGAN2-ada-Generator':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1EsGehuEdY4z4t21o2LgW2dSsyN3rxYLJ',
                    'dir': './ckpts',
                    'file_type': 'pth',
                    'mode': 'gdown'
                },
            'e4e_ffhq_encode':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1cUv_reLE6k3604or78EranS7XzuVMWeO',
                    'dir': './ckpts',
                    'file_type': 'pt',
                    'mode': 'gdown'
                },
            'model_ir_se50':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1GIMopzrt2GE_4PG-_YxmVqTQEiaqu5L6',
                    'dir': './ckpts',
                    'file_type': 'pth',
                    'mode': 'gdown'
                },
            'face_parsing':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1IMsrkXA9NuCEy1ij8c8o6wCrAxkmjNPZ',
                    'dir': './ckpts',
                    'file_type': 'pth',
                    'mode': 'gdown'
                },
            'vgg16':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1EPhkEP_1O7ZVk66aBeKoFqf3xiM4BHH8',
                    'dir': './ckpts',
                    'file_type': 'pth',
                    'mode': 'gdown'
                },
            'classification_model_gender':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1SSw6vd-25OGnLAE0kuA-_VHabxlsdLXL',
                    'dir': './classifier/gender_classification',
                    'file_type': 'pth',
                    'mode': 'gdown'
                },
            'classification_model_hair':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1n14ckDcgiy7eu-e9XZhqQYb5025PjSpV',
                    'dir': './classifier/hair_classification',
                    'file_type': 'pth',
                    'mode': 'gdown'
                },
            'hair_mapper':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1F3oujXbvalqEOixcAkIyURuY512nmroe',
                    'dir': './mapper/checkpoints/final',
                    'file_type': 'pt',
                    'mode': 'gdown'
                }
        }

    def _run(self, *img_paths):
        """
        Implementation of the private run method of the base class. Runs the hair mapper model and saves the image using
        cv2.

        :param img_paths: input image path(s)
        :return: result image path
        """
        img_path = img_paths[0]

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # baldify image
        res_img = self._hair_mapper(img_path)

        # save image
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        res_img_path = os.path.join(self._tmp_dir, f"{img_name}_{self._get_img_suffix()}{img_ext}")

        cv2.imwrite(res_img_path, res_img)

        return res_img_path

    def _save_result(self, img_name, res):
        """
        Saves the result using the cv2 library.

        :param img_name: image name
        :param res: result image
        :return: result image name
        """
        img_name = img_name.replace(".png", f"_{self._get_img_suffix()}.png")

        res_path = os.path.join(self._tmp_dir, img_name)
        cv2.imwrite(res_path, res)

        return img_name

    def _encode(self, img_path):
        """
        This method translates image data to latent code data using the FFHQ encoder.

        :param img_path: image path
        :return: image in latent code
        """

        # pre-define image transforms
        img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # set up model for encoding
        encoder_path = "./ckpts/e4e_ffhq_encode.pt"
        ckpt = torch.load(encoder_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = encoder_path
        opts = Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()

        # open image from path
        input_img = PIL.Image.open(img_path)

        # transform image
        transformed_img = img_transforms(input_img)

        # encode image
        with torch.no_grad():
            latents = self._run_on_batch(transformed_img.unsqueeze(0), net)
            latent = latents[0].cpu().numpy()
            latent = np.reshape(latent, (1, 18, 512))

        return latent

    def _run_on_batch(self, inputs, net):
        """
        Feeds input into a network in batches.

        :param inputs: input file(s)
        :param net: network
        :return: result after feeding input into network
        """
        result = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        return result

    def _hair_mapper(self, img_path):
        """
        This method runs the HairMapper model according to the provided Jupyter Notebook in references/HairMapper.
        The input image is first translated into latent code using an encoder. The latent code is then fed through the
        HairMapper network which uses StyleGAN2 as a generator. The result is a baldified image with the same face
        identity as the input image.

        :param img_path: input image path
        :return: bald image
        """

        # run encoder
        latent_code = self._encode(img_path)

        # set up generator
        model_name = 'stylegan2_ada'
        latent_space_type = 'wp'

        # initialize generator
        print(f'Initializing Hair Mapper Generator ...')
        print(model_name)
        model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=1.0)

        # set up Hair Mapper model
        mapper = LevelMapper(input_dim=512).eval().cuda()

        # get path to model
        print(os.getcwd())
        hair_mapper_path = self._get_pretrained_model_path('hair_mapper')

        # load model
        ckpt = torch.load(hair_mapper_path)
        alpha = float(ckpt['alpha']) * 1.2
        mapper.load_state_dict(ckpt['state_dict'], strict=True)
        kwargs = {'latent_space_type': latent_space_type}

        # set up parsing network
        parsing_net_path = self._get_pretrained_model_path('face_parsing')
        parsing_net = get_parsingNet(save_pth=parsing_net_path)

        # set up latent code as input for hair mapper
        latent_code_origin = np.reshape(latent_code, (1, 18, 512))

        mapper_input = latent_code_origin.copy()
        mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
        edited_latent_codes = latent_code_origin
        edited_latent_codes[:, :8, :] += alpha * mapper(mapper_input_tensor).to('cpu').detach().numpy()

        # set up model for style mixing
        outputs = model.easy_style_mixing(
            latent_codes=edited_latent_codes,
            style_range=range(7, 18),
            style_codes=latent_code_origin,
            mix_ratio=0.8,
            **kwargs
        )

        # edit image using style mixing model
        edited_img = outputs['image'][0][:, :, ::-1]

        # get origin image
        origin_img = cv2.imread(img_path)

        # get hair mask and edit
        hair_mask = get_hair_mask(img_path=origin_img, net=parsing_net, include_hat=True, include_ear=True)

        mask_dilate = cv2.dilate(hair_mask, kernel=np.ones((50, 50), np.uint8))
        mask_dilate_blur = cv2.blur(mask_dilate, ksize=(30, 30))
        mask_dilate_blur = (hair_mask + (255 - hair_mask) / 266 * mask_dilate_blur).astype(np.uint8)

        face_mask = 255 - mask_dilate_blur

        index = np.where(face_mask > 0)
        cy = (np.min(index[0]) + np.max(index[0])) // 2
        cx = (np.min(index[1]) + np.max(index[1])) // 2
        center = (cx, cy)

        mixed_clone = cv2.seamlessClone(origin_img, edited_img, face_mask[:, :, 0], center, cv2.NORMAL_CLONE)

        return mixed_clone
