import bz2

import PIL.Image
import requests
import os
from bz2 import decompress


def load_image(img_path, downsample=False):
    """loads image from path"""
    # get rgb image from image path
    img = PIL.Image.open(img_path).convert("RGB")

    # downsample image
    if downsample:
        img = img.resize((256, 256))

    return img


def save_image(img, name, output_dir):
    """saves image into output folder"""
    # create output folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # save image to output folder
    img.save(os.path.join(output_dir, f"{name}.png"), "PNG")


def cli_progress_bar_download(dl, total_size, width=50):
    """displays download progress"""
    progress = dl / total_size if total_size > 0 else 0
    filled_width = int(width * progress)
    progress_bar_str = '[' + '█' * filled_width + '░' * (width - filled_width) + ']'
    print(f"Progress: {progress_bar_str} {progress * 100:.2f}%", end="\r")


def load_pretrained_model(url, name, file_ending):
    """loads pretrained model from network pickle"""
    # get current path
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    # setup destination folder and path
    dest_dir = os.path.join(curr_dir, "../..", "pretrained_models")
    os.makedirs(dest_dir, exist_ok=True)
    dest_dir = os.path.join(dest_dir, f"{name}.{file_ending}")

    # download file if it not exists
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
                    cli_progress_bar_download(dl, total_size)

        # unzip bz2 file
        if file_ending == "dat.bz2":
            decompress_bz2(dest_dir)

        # check success
        if os.path.exists(dest_dir):
            print(f"\nDownload {name} complete!")
        else:
            print(f"Download {name} failed!")


def decompress_bz2(file_path):
    zipfile = bz2.BZ2File(file_path)  # open the file
    data = zipfile.read()  # get the decompressed data
    new_file_path = file_path[:-4]  # assuming the filepath ends with .bz2
    open(new_file_path, 'wb').write(data)
