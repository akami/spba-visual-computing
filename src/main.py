from models.hair_editor import HairEditor
from models.preprocessing_models.baldification import Baldification
from models.preprocessing_models.ffhq_aligner import FfhqAligner
import os
import argparse
from util.model_loader import ModelLoader


def main():
    # parse arguments
    args = parse_args()

    # working paths
    # TODO root path may not be project path
    root = os.getcwd()
    tmp_dir = args.tmp_dir if os.path.isabs(args.tmp_dir) else os.path.join(root, args.tmp_dir)

    # create temp directory
    os.makedirs(tmp_dir, exist_ok=True)

    # model paths
    aligner_dir = root
    baldification_dir = os.path.join(root, "references/HairMapper")
    hair_editor = root

    # image paths
    face_img_path = args.face_img if os.path.isabs(args.face_img) else os.path.join(root, args.face_img)
    hair_img_path = args.hair_img if os.path.isabs(args.hair_img) else os.path.join(root, args.hair_img)

    # TODO implement usage
    res_img_path = args.res_img if os.path.isabs(args.res_img) else os.path.join(root, args.res_img)

    # instantiate model loader
    loader = ModelLoader()

    # defining models
    aligner = FfhqAligner(loader, aligner_dir, tmp_dir)
    baldification = Baldification(loader, baldification_dir, tmp_dir)
    hair_editor = HairEditor(loader, hair_editor, tmp_dir)

    # preprocess face image
    if args.face_align:
        face_img_path = aligner.run(face_img_path)

    # preprocess hair image
    if args.hair_align:
        hair_img_path = aligner.run(hair_img_path)

    hair_img_path = baldification.run(hair_img_path)

    # combine images
    # TODO implement
    # res_img_path = hair_editor.run(face_img_path, hair_img_path)


def parse_args():
    """parses command line arguments"""
    parser = argparse.ArgumentParser(prog="SPBA", description="Hair Editing with Generative Adversarial Networks")

    # set default input and output folders
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="The directory where temporary images are stored")

    # image arguments
    parser.add_argument("--face_img", type=str, default="hair.png", help="Path to face identity image")
    parser.add_argument("--face_align", type=bool, default=False, help="Align face identity image")

    parser.add_argument("--hair_img", type=str, default="face.png", help="Path to hair identity image")
    parser.add_argument("--hair_align", type=bool, default=False, help="Align hair identity image")

    parser.add_argument("--res_img", type=str, default="result.png", help="Path to where the result image is stored")

    return parser.parse_args()


if __name__ == "__main__":
    main()
