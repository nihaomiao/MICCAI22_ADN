# Test alignment network, i.e., the Transformation network in our paper
import os
import torch
import numpy as np
import cv2
from PIL import Image
from model.transform_net import PlaneFinder
import nibabel as nib
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import imageio


def max_min_norm(x):
    b, c, d, h, w = x.size()
    x_min = x.view(b, c, d, -1).min(dim=-1)[0].view(b, c, d, 1, 1).repeat(1, 1, 1, h, w)
    x_max = x.view(b, c, d, -1).max(dim=-1)[0].view(b, c, d, 1, 1).repeat(1, 1, 1, h, w)
    norm_x = (x - x_min)/(x_max - x_min)
    return norm_x


def main():
    useless_label = [4]
    save_dir = "/data/StrokeCT/align_net/diff_gif"
    msk_size = 256
    os.makedirs(save_dir, exist_ok=True)
    # load model
    model = PlaneFinder(is_train=False)
    model = model.cuda()
    trained_model_pth = "/data/StrokeCT/align_net/snapshots/B0040_S012500.pth"
    checkpoint = torch.load(trained_model_pth)
    for name, _ in checkpoint['state_dict'].items():
        model.state_dict()[name].copy_(checkpoint['state_dict'][name])
    model.cuda()
    model.eval()

    # load data
    num_slice = 40
    # directly evaluate the whole dataset
    ct_dir_path = "/data/StrokeCT/AISD_data_resample"
    ct_name_list = os.listdir(ct_dir_path)
    ct_name_list.sort()
    for ct_name in ct_name_list:
        print(ct_name)
        ct_path = os.path.join(ct_dir_path, ct_name, "CT.nii")
        msk_path = os.path.join(ct_dir_path, ct_name, "mask.nii")
        msk_data = nib.load(msk_path).get_fdata()

        ct_name = ct_path.split("/")[-2]
        ct_data = nib.load(ct_path).get_fdata()
        chosen_ct_tensor = transforms.ToTensor()(ct_data)
        chosen_ct_tensor = chosen_ct_tensor/255.0
        chosen_ct_tensor = chosen_ct_tensor.unsqueeze(dim=1)
        chosen_ct_tensor = F.rotate(chosen_ct_tensor, -90)
        chosen_ct_tensor = F.resize(chosen_ct_tensor, size=[256, 256])
        chosen_ct_tensor = chosen_ct_tensor.permute(1, 0, 2, 3)
        chosen_ct_tensor = chosen_ct_tensor.unsqueeze(dim=0)
        chosen_ct_tensor = chosen_ct_tensor.cuda()

        images_t, images_r, _, view, _, _ = model(chosen_ct_tensor.type(torch.float32))

        # compute difference maps
        chosen_ct_tensor = max_min_norm(chosen_ct_tensor)
        images_r = max_min_norm(images_r)
        d = images_r - chosen_ct_tensor
        d[d < 0] = 0
        d[d == images_r] = 0
        d[d == chosen_ct_tensor] = 0
        d = torch.log10(1+1000*d)/3.0

        new_image_list = []
        # visualize data
        for ii in range(num_slice):
            cur_image = 255 * chosen_ct_tensor[0, 0, ii, :, :].data.cpu().numpy()
            cur_image_r = 255 * images_r[0, 0, ii, :, :].data.cpu().numpy()
            cur_image_d = 255 * d[0, 0, ii, :, :].data.cpu().numpy()

            cur_image = np.asarray(cur_image, dtype=np.uint8)
            cur_image_r = np.asarray(cur_image_r, dtype=np.uint8)
            cur_image_d = np.asarray(cur_image_d, dtype=np.uint8)

            gt_now = msk_data[:, :, ii]
            gt_now = cv2.rotate(gt_now, cv2.ROTATE_90_CLOCKWISE)
            # deal with useless label
            # deal with special value 4
            useless_area = np.zeros_like(gt_now, dtype=np.bool)
            for ll in useless_label:
                useless_area = np.logical_or(useless_area, gt_now == ll)
            gt_now[useless_area] = 0
            # consider value 1, 2, 3, 5 as infarct areas
            infarct_area = gt_now != 0
            gt_now[infarct_area] = 1
            gt_now = gt_now * 255.0
            cur_img_show = np.array(cur_image, dtype=np.uint8)
            cur_img_show = np.repeat(np.expand_dims(cur_img_show, axis=2), 3, axis=2)
            cur_lbl_show = np.zeros((msk_size, msk_size, 3))
            cur_lbl_show[:, :, 0] = gt_now
            cur_img_with_lbl = 0.7*np.array(cur_img_show, dtype=np.float32) + 0.3*np.array(cur_lbl_show, dtype=np.float32)
            bg_msk = np.repeat(np.expand_dims(gt_now == 0, axis=2), 3, axis=2)
            cur_img_with_lbl[bg_msk] = cur_img_show[bg_msk]
            cur_img_with_lbl = np.array(cur_img_with_lbl, dtype=np.uint8)

            new_image = Image.new('RGB', (msk_size*4, msk_size))
            new_image.paste(Image.fromarray(cur_image, "L"), (0, 0))
            new_image.paste(Image.fromarray(cur_image_r, "L"), (msk_size, 0))
            new_image.paste(Image.fromarray(cur_image_d, "L"), (msk_size * 2, 0))
            new_image.paste(Image.fromarray(cur_img_with_lbl, "RGB"), (msk_size * 3, 0))
            new_image_list.append(np.asarray(new_image))
        # save gif
        new_image_path = os.path.join(save_dir, ct_name + ".gif")
        imageio.mimsave(new_image_path, new_image_list)
