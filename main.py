from src.util import util
import os
import argparse
from src.util.model_loader import ModelLoader
from src.preprocessing.Preprocessing import Preprocessing

def main():
    # parse arguments
    args = parse_args()

    # load pretrained models
    model_loader = ModelLoader()
    model_loader.load_models()

    # get image paths
    img_path1 = os.path.join(args.input_dir, args.img1)
    img_path2 = os.path.join(args.input_dir, args.img2)

    # load input images
    img1 = util.load_image(img_path1)
    img2 = util.load_image(img_path2)

    # preprocess image
    preprocessing = Preprocessing()
    aligned_image_identity = preprocessing.preprocess(img_path1, in_the_wild=False)
    aligned_image_hair = preprocessing.preprocess(img_path2, in_the_wild=True)


    # save images images in output folder
    util.save_image(img1, "img1", args.output_dir)
    util.save_image(img2, "img2", args.output_dir)
    util.save_image(aligned_image_identity, "img1_preprocessed", args.output_dir)
    util.save_image(aligned_image_hair, "img2_preprocessed", args.output_dir)


def parse_args():
    """parses command line arguments"""
    parser = argparse.ArgumentParser(prog="SPBA", description="Hair Editing with Generative Adversarial Networks")

    # set default input and output folders
    parser.add_argument("--input_dir", type=str, default="input", help="The directory of the input images")
    parser.add_argument("--output_dir", type=str, default="output", help="The directory to store the results")

    # set default images
    parser.add_argument("--img1", type=str, default="00059.png", help="Face Identity Image")
    parser.add_argument("--img2", type=str, default="00275.png", help="Hair Identity Image")

    return parser.parse_args()


if __name__ == "__main__":
    main()
