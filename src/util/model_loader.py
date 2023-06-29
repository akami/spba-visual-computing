import bz2
import requests
import os
from src.util import util
import gdown

model_dic = {
    'ffhq':
        {
            'url': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
            'dir': './pretrained_models',
            'file_type': 'pkl'
        },
    "dlib_shape_predictor":
        {
            'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'dir': './pretrained_models',
            'file_type': 'dat.bz2'
        },
    # ==================== Hair Mapper Baldification Models ====================
    'StyleGAN2-ada-Generator':
        {
            'url': 'https://drive.google.com/file/d/1EsGehuEdY4z4t21o2LgW2dSsyN3rxYLJ',
            'dir': './pretrained_models/hair_mapper_models/ckpts',
            'file_type': 'pth'
        },
    'e4e_ffhq_encode':
        {
            'url': 'https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO',
            'dir': './pretrained_models/hair_mapper_models/ckpts',
            'file_type': 'pt'
        },
    'model_ir_se50':
        {
            'url': 'https://drive.google.com/file/d/1GIMopzrt2GE_4PG-_YxmVqTQEiaqu5L6',
            'dir': './pretrained_models/hair_mapper_models/ckpts',
            'file_type': 'pth'
        },
    'face_parsing':
        {
            'url': 'https://drive.google.com/file/d/1IMsrkXA9NuCEy1ij8c8o6wCrAxkmjNPZ',
            'dir': './pretrained_models/hair_mapper_models/ckpts',
            'file_type': 'pth'
        },
    'vgg16':
        {
            'url': 'https://drive.google.com/file/d/1EPhkEP_1O7ZVk66aBeKoFqf3xiM4BHH8',
            'dir': './pretrained_models/hair_mapper_models/ckpts',
            'file_type': 'pth'
        },
    'classification_model_gender':
        {
            'url': 'https://drive.google.com/file/d/1SSw6vd-25OGnLAE0kuA-_VHabxlsdLXL',
            'dir': './pretrained_models/hair_mapper_models/classifier/gender_classification',
            'file_type': 'pth'
        },
    'classification_model_hair':
        {
            'url': 'https://drive.google.com/file/d/1n14ckDcgiy7eu-e9XZhqQYb5025PjSpV',
            'dir': './pretrained_models/hair_mapper_models/classifier/hair_classification',
            'file_type': 'pth'
        },
    'hair_mapper':
        {
            'url': 'https://drive.google.com/file/d/1F3oujXbvalqEOixcAkIyURuY512nmroe',
            'dir': './pretrained_models/hair_mapper_models',
            'file_type': 'pt'
        }
}


class ModelLoader:

    def load_models(self):
        # get current path
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        for model in model_dic:

            # get info from model dictionary
            name = model  # key is the name
            url = model_dic[name]['url']  # url
            dest_dir = model_dic[name]['dir']  # destination directory
            file_ending = model_dic[name]['file_type']  # file ending

            # setup destination folder and path
            dest_dir = os.path.join(curr_dir, "../..", dest_dir)
            os.makedirs(dest_dir, exist_ok=True)
            dest_dir = os.path.join(dest_dir, f"{name}.{file_ending}")

            # download file only if it not exists yet
            if not os.path.exists(dest_dir):

                # download file
                response = requests.get(url, stream=True)

                # display progress
                total_size = int(response.headers.get("content-length", 0))
                with open(dest_dir, "wb") as f:
                    print(f"Downloading {name} ... \n")
                    dl = 0
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            dl += len(chunk)
                            f.write(chunk)
                            util.cli_progress_bar_download(dl, total_size)

                # check success
                if os.path.exists(dest_dir):
                    print(f"\nDownload {name} complete!")
                else:
                    print(f"Download {name} failed!")

                # unzip bz2 files
                if file_ending == "dat.bz2":
                    self.decompress_bz2(dest_dir)

    def decompress_bz2(self, file_path):
        zipfile = bz2.BZ2File(file_path)  # open the file
        data = zipfile.read()  # get the decompressed data
        new_file_path = file_path[:-4]  # assuming the filepath ends with .bz2
        open(new_file_path, 'wb').write(data)
