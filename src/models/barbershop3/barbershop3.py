import os.path

from models.model import Model
from models.barbershop3.models.alignment.segmentation import Segmentation


# TODO documentation
# TODO implementation
class Barbershop3(Model):
    def __init__(self, loader, model_dir, tmp_dir):
        super().__init__(loader, model_dir, tmp_dir)

    def _get_img_suffix(self):
        return "barbershop3"

    def _get_pretrained_models(self):
        return {
            # Segmentation
            'Segmentation-Network':
                {
                    'url': 'https://drive.google.com/u/0/uc?id=1lIKvQaFKHT5zC7uS4p17O9ZpfwmwlS62',
                    'dir': './pretrained_models',
                    'file_type': 'pth',
                    'mode': 'gdown'
                }
        }

    def _setup(self):
        # download pretrained models using super method
        super()._setup()

        #### SETUP Segmentation Network ####
        # set model path
        attrs = self._get_pretrained_models()['Segmentation-Network']
        model_path = os.path.join(attrs['dir'], f"Segmentation-Network.{attrs['file_type']}")
        os.makedirs(attrs['dir'], exist_ok=True)

        # set tmp dir path
        tmp_dir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        # initialize Segmentation class
        self._segmentation = Segmentation(model_path, tmp_dir)

    # TODO implement
        # set up embedding step
        # self.ii2s_embedder = Embedding

        # set up alignment step
        # self.mask_aligner = Alignment

        # set up image blending
        # self.img_blender = Blending

    def _run(self, *img_paths):
        face_identity_img_path = img_paths[0]
        hair_identity_img_path = img_paths[1]

        self._segmentation.run(face_identity_img_path, hair_identity_img_path)

    # TODO implement

        # save image
        # TODO use dict as input and return type for run methods in model subclass; document within implementations of run methods
        # img_name, img_ext = os.path.splitext(os.path.basename(face_identity_img_path))
        # res_img_path = os.path.join(self._tmp_dir, f"{img_name}_{self._get_img_suffix()}{img_ext}")

        # res_img.save(res_img_path)

        # return res_img_path
        return ""
