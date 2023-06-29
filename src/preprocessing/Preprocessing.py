import PIL
import PIL.Image
import dlib

from src.preprocessing import align_in_the_wild as align


class Preprocessing:
    def __init__(self):
        # set up predictor model
        self.predictor = dlib.shape_predictor("./pretrained_models/dlib_shape_predictor.dat")

    def preprocess(self, img_file_path, in_the_wild=False):
        img = PIL.Image.open(img_file_path)

        if in_the_wild:     # in-the-wild images need to be translated/aligned according to FFHQ data
            # get landmarks
            dlib_img = dlib.load_rgb_image(img_file_path)
            lm = align.get_landmark(dlib_img, self.predictor)

            # align face
            aligned_img = align.align_face(lm, img)
            img = aligned_img

        return img

