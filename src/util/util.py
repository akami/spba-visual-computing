import PIL.Image
import os


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
