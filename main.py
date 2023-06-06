from util import util
import os
import argparse

model_dic = {"ffhq": "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl"}
def main():
    # parse arguments
    args = parse_args()

    # load pretrained model
    util.load_pretrained_model(model_dic["ffhq"], "ffhq")

    # get image paths
    img_path1 = os.path.join(args.input_dir, args.img1)
    img_path2 = os.path.join(args.input_dir, args.img2)

    # load and downsample images
    img1_downsampled = util.load_image(img_path1, downsample=True)
    img2_downsampled = util.load_image(img_path2, downsample=True)

    # save downsampled images in output folder
    util.save_image(img1_downsampled, "img1_downsampled", args.output_dir)
    util.save_image(img2_downsampled, "img2_downsampled", args.output_dir)


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
