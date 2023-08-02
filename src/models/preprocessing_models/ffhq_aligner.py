import os.path

import numpy as np
import scipy
import PIL
import PIL.Image
import dlib

from models.model import Model


"""
The following methods borrow heavily from Barbershop reference: references/Barbershop/utils/shape_predictor.py:

############# COPY START #############
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
############# COPY END #############
"""


class FfhqAligner(Model):
    """
    This class represents the preprocessing-step in which in-the-wild pictures are aligned according to the FFHQ dataset.
    The FFHQ dataset was used to train the StyleGAN model. So aligning pictures that do not come from this dataset is
    crucial in making sure that the resulting picture looks realistic, i.e. does not show unwanted artifacts.
    In particular, the portrait picture is cropped and centered around the face. We use dlib landmark detection to align
    and match the position of facial landmarks according to FFHQ.
    Only use this model as a preprocessing-step if your input portrait picture is in fact in-the-wild. Otherwise,
    processing already aligned pictures will result in blurring and/or unwanted artifacts.
    """
    def __init__(self, loader, model_dir, tmp_dir):
        super().__init__(loader, model_dir, tmp_dir)

    def _get_img_suffix(self):
        return "aligned"

    def _get_pretrained_models(self):
        return {
            'ffhq':
                {
                    'url': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
                    'dir': './pretrained_models',
                    'file_type': 'pkl',
                    'mode': 'default'
                },
            "dlib_shape_predictor":
                {
                    'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
                    'dir': './pretrained_models',
                    'file_type': 'dat.bz2',
                    'mode': 'default'
                }
        }

    def _setup(self):
        super()._setup()

        predictor_path = "./pretrained_models/dlib_shape_predictor.dat"

        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)

    def _run(self, *img_paths):
        img_path = img_paths[0]

        # get landmarks
        dlib_img = dlib.load_rgb_image(img_path)
        lm = self._get_landmark(dlib_img)

        # align face
        img = PIL.Image.open(img_path)
        res_img = self._align_face(lm, img)

        # save image
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        res_img_path = os.path.join(self._tmp_dir, f"{img_name}_{self._get_img_suffix()}{img_ext}")

        res_img.save(res_img_path, "PNG")

        return res_img_path

    def _align_face(self, lm, img, output_size=1024):
        """
        This method aligns in-the-wild pictures according to the FFHQ dataset in order for the network
        to recognize facial features and to minimize artifacts due to misalignment.

        :param lm: image landmarks
        :param img: RGB image file path
        :param output_size: image size
        :return: image translated into FFHQ format
        """

        ############# COPY START #############

        # Pre-define landmark data.
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        ############# COPY END #############

        transform_size = 4096
        enable_padding = True

        ############# COPY START #############

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        ############# COPY END #############

        return img

    def _get_landmark(self, img):
        """
        This method detects landmarks in a face using the dlib shape predictor.
        :param img: input image
        :return: array of facial landmarks, shape=(68, 2)
        """

        # set up detector
        detector = dlib.get_frontal_face_detector()
        detected = detector(img, 1)

        assert len(detected) > 0, "Face not detected!"

        for k, d in enumerate(detected):
            shape = self.predictor(img, d)

        landmark = np.array(
            [
                [pixel.x, pixel.y] for pixel in shape.parts()
            ]
        )

        return landmark
