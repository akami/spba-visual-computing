import PIL.Image
import os

def load_image(img_path, downsample=True):
    img = PIL.Image.open(img_path).convert("RGB")
    if downsample:
        img = img.resize((256, 256))
    return img


def save_image(img, name, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    img.save(os.path.join(output_dir, f"{name}.png"), "PNG")
