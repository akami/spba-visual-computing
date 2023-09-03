from PIL import Image
import numpy as np
import torchvision
import os


def save_tensor_to_img(tensor, img_name, img_ext, dest_dir, suffix="tmp"):
    toPIL = torchvision.transforms.ToPILImage()
    res_img = toPIL(((tensor[0] + 1) / 2).detach().cpu().clamp(0, 1))

    res_img_path = os.path.join(dest_dir, f"{img_name}_{suffix}{img_ext}")
    res_img.save(res_img_path)


def set_segmentation_colors(tensor):
    num_labels = 16

    color = np.array([[0, 0, 0],  ## 0
                      [102, 204, 255],  ## 1
                      [255, 204, 255],  ## 2
                      [255, 255, 153],  ## 3
                      [255, 255, 153],  ## 4
                      [255, 255, 102],  ## 5
                      [51, 255, 51],  ## 6
                      [0, 153, 255],  ## 7
                      [0, 255, 255],  ## 8
                      [0, 255, 255],  ## 9
                      [204, 102, 255],  ## 10
                      [0, 153, 255],  ## 11
                      [0, 255, 153],  ## 12
                      [0, 51, 0],
                      [102, 153, 255],  ## 14
                      [255, 153, 102],  ## 15
                      ])

    h, w = np.shape(tensor)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     print(color.shape)

    # TODO what does this do?
    for ii in range(num_labels):
        #         print(ii)
        mask = tensor == ii
        rgb[mask, None] = color[ii, :]

    # TODO what does this do?
    # Correct unk
    unk = tensor == 255
    rgb[unk, None] = color[0, :]

    return rgb

def save_seg_mask_to_img(img_name, dest_dir, mask_tensor):
    vis_path = os.path.join(dest_dir, 'segmentation_mask_{}.png'.format(img_name))
    vis_mask = set_segmentation_colors(mask_tensor)
    Image.fromarray(vis_mask).save(vis_path)
