# test ADN
import argparse
import numpy as np
import time
import torch
from torch.utils import data
from dataset import StrokeTest3D
from model.unet3d.unet_model import ResidualUNet3D
from model.transform_net import PlaneFinder
import os
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import sys
from misc import Logger, map2fig
import timeit
from scipy.sparse import save_npz, coo_matrix
import imageio
from tqdm import tqdm


start = timeit.default_timer()
useless_label = [4]
GPU = "0"
postfix = "-adn-wl1.0"
BATCH_SIZE = 10
NUM_EPOCH = 100
NUM_SLICE = 40
INPUT_SIZE = [256, 256]
root_dir = "/data/StrokeCT/adn"
data_dir = "/data/StrokeCT/AISD_data_resample"
test_txt = "/data/StrokeCT/aisd_test.txt"
NUM_CLASSES = 2
RANDOM_SEED = 1234
# put pretrained model D and F here
RESTORE_FROM = "B0006_S050000.pth"
# put pretrained model T here
ALIGN_RESTORE_FROM = "B0040_S012500.pth"
MSK_PATH = os.path.join(root_dir, "ckpt_"+str(NUM_EPOCH)+postfix, "msk")
os.makedirs(MSK_PATH, exist_ok=True)
# probability map
MAP_PATH = os.path.join(root_dir, "ckpt_"+str(NUM_EPOCH)+postfix, "map")
os.makedirs(MAP_PATH, exist_ok=True)
SHOW_PATH = os.path.join(root_dir, "ckpt_"+str(NUM_EPOCH)+postfix, "show")
os.makedirs(SHOW_PATH, exist_ok=True)
GIF_PATH = os.path.join(root_dir, "ckpt_"+str(NUM_EPOCH)+postfix, "gif")
os.makedirs(GIF_PATH, exist_ok=True)
LOG_PATH = os.path.join(root_dir, "ckpt_"+str(NUM_EPOCH)+postfix, "ckpt_"+str(NUM_EPOCH)+postfix+".log")
sys.stdout = Logger(LOG_PATH, sys.stdout)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UNet Network")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--align-restore-from", type=str, default=ALIGN_RESTORE_FROM)
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print frequency')
    return parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = get_arguments()
    sys.stdout = Logger(args.log_path, sys.stdout, mode="w")
    args.postfix = "-" + args.postfix
    use_norm = "norm" in args.postfix
    use_dn = "dn" in args.postfix
    print("use dn", use_dn)
    print("use norm", use_norm)
    if use_dn:
        data_dir = "/data/hfn5052/StrokeCT/AISD_data_resample_denoise_mct_seg"
    else:
        data_dir = "/data/hfn5052/StrokeCT/AISD_data_resample"
    print(data_dir)
    MSK_PATH = os.path.join(args.root_dir, "ckpt_"+str(args.num_epoch)+args.postfix, "msk")
    os.makedirs(MSK_PATH, exist_ok=True)
    # probability map
    MAP_PATH = os.path.join(args.root_dir, "ckpt_"+str(args.num_epoch)+args.postfix, "map")
    os.makedirs(MAP_PATH, exist_ok=True)
    SHOW_PATH = os.path.join(args.root_dir, "ckpt_"+str(args.num_epoch)+args.postfix, "show")
    os.makedirs(SHOW_PATH, exist_ok=True)
    ASYM_PATH = os.path.join(args.root_dir, "ckpt_"+str(args.num_epoch)+args.postfix, "asym")
    os.makedirs(ASYM_PATH, exist_ok=True)
    GIF_PATH = os.path.join(args.root_dir, "ckpt_"+str(args.num_epoch)+args.postfix, "gif")
    os.makedirs(GIF_PATH, exist_ok=True)
    print(args.postfix)
    print(MSK_PATH)
    print("Restored from:", args.restore_from)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load alignment model
    align_model = PlaneFinder(is_train=False)
    align_model.cuda()

    saved_align_state_dict = torch.load(args.align_restore_from)
    align_model.load_state_dict(saved_align_state_dict['state_dict'])
    align_model.eval()

    seg_model = ResidualUNet3D(in_channels=1, out_channels=1, f_maps=32, use_transconv=False)
    asym_model = ResidualUNet3D(in_channels=1, out_channels=1, f_maps=32, use_transconv=False)
    seg_model = torch.nn.DataParallel(seg_model)
    asym_model = torch.nn.DataParallel(asym_model)
    asym_model.cuda()
    seg_model.cuda()

    saved_state_dict = torch.load(args.restore_from)
    seg_model.load_state_dict(saved_state_dict['seg_state_dict'])
    asym_model.load_state_dict(saved_state_dict['asym_state_dict'])
    asym_model.eval()
    seg_model.eval()

    test_loader = data.DataLoader(StrokeTest3D(data_dir=data_dir, test_txt=test_txt),
                                  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    batch_time = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for index, (images, labels, names) in enumerate(test_loader):
            # deal with special value 4
            useless_area = torch.zeros_like(labels, dtype=torch.bool)
            for ll in useless_label:
                useless_area = torch.logical_or(useless_area, labels == ll)
            labels[useless_area] = 0
            # consider value 1, 2, 3, 5 as infarct areas
            infarct_area = labels != 0
            labels[infarct_area] = 1
            msk_size = images.shape[-1]
            images = images.type(torch.float32)
            images = images.cuda()
            empty_mask = images == 0

            images_t, _, images_t_f, _, M, M_inv = align_model(images)
            diff_t = images_t - images_t_f
            sym_comp_t = torch.zeros_like(images_t)
            sym_comp_t[diff_t > 0] = images_t[diff_t > 0]
            sym_comp_t[diff_t == 0] = images_t[diff_t == 0]
            sym_comp_t[diff_t < 0] = images_t_f[diff_t < 0]
            asym_map_t = nn.ReLU()(images_t_f - images_t)

            subject_asym_conf_t = asym_model(images_t)
            anatomy_asym_conf_t = asym_map_t - subject_asym_conf_t
            anatomy_asym_conf_t = nn.ReLU()(anatomy_asym_conf_t)
            subject_asym_images_t = images_t + anatomy_asym_conf_t

            anatomy_asym_images_t = images_t + subject_asym_conf_t
            anatomy_asym_images_t = torch.clamp(anatomy_asym_images_t, max=sym_comp_t)

            preds_t = seg_model(subject_asym_images_t)
            preds = stn(preds_t, M_inv[:, :3, :])
            preds = preds.squeeze(dim=1)
            preds[useless_area] = 0
            preds[empty_mask.squeeze(dim=1)] = 0

            # save all results
            for b in range(0, NUM_SLICE):
                for ind in range(0, preds.size(0)):
                    ct_name = names[ind].split("_")[0]
                    cur_map_path = os.path.join(MAP_PATH, ct_name)
                    cur_msk_path = os.path.join(MSK_PATH, ct_name)
                    cur_show_path = os.path.join(SHOW_PATH, ct_name)
                    cur_asym_path = os.path.join(ASYM_PATH, ct_name)
                    os.makedirs(cur_map_path, exist_ok=True)
                    os.makedirs(cur_msk_path, exist_ok=True)
                    os.makedirs(cur_show_path, exist_ok=True)
                    os.makedirs(cur_asym_path, exist_ok=True)
                    prob = preds[ind, b, :, :].data.cpu().numpy()
                    heatmap = map2fig(prob)
                    prob = coo_matrix(prob)
                    if len(prob.data) == 0:
                        continue
                    map_name = names[ind] + "_%03d.npz" % b
                    map_file = os.path.join(cur_map_path, map_name)
                    save_npz(map_file, prob.tocsr())

                    msk = preds[ind, b, :, :].data.cpu().numpy() > 0.5
                    msk = msk * 255.0
                    msk_name = names[ind] + "_%03d.png" % b
                    msk_file = os.path.join(cur_msk_path, msk_name)
                    msk_img = Image.fromarray(msk.astype("uint8"), "L")
                    msk_img.save(msk_file)

                    img_now = images[ind, :, b, :, :].squeeze().data.cpu().numpy()
                    img_now = img_now * 255.0
                    gt_now = labels[ind, b, :, :].squeeze().data.cpu().numpy()
                    gt_now = gt_now * 255.0
                    # mix image with label
                    cur_img_show = np.array(img_now, dtype=np.uint8)
                    cur_img_show = np.repeat(np.expand_dims(cur_img_show, axis=2), 3, axis=2)
                    cur_lbl_show = np.zeros((msk_size, msk_size, 3))
                    cur_lbl_show[:, :, 0] = gt_now
                    cur_img_with_lbl = 0.7*np.array(cur_img_show, dtype=np.float32) + 0.3*np.array(cur_lbl_show, dtype=np.float32)
                    bg_msk = np.repeat(np.expand_dims(gt_now == 0, axis=2), 3, axis=2)
                    cur_img_with_lbl[bg_msk] = cur_img_show[bg_msk]
                    cur_img_with_lbl = np.array(cur_img_with_lbl, dtype=np.uint8)
                    # mix image with pred
                    cur_pred_show = np.zeros((msk_size, msk_size, 3))
                    cur_pred_show[:, :, 0] = msk
                    cur_img_with_pred = 0.7*np.array(cur_img_show, dtype=np.float32) + 0.3*np.array(cur_pred_show, dtype=np.float32)
                    cur_pred_msk = np.repeat(np.expand_dims(msk == 0, axis=2), 3, axis=2)
                    cur_img_with_pred[cur_pred_msk] = cur_img_show[cur_pred_msk]
                    cur_img_with_pred = np.array(cur_img_with_pred, dtype=np.uint8)
                    # mix image with heatmap
                    cur_img_with_heatmap = 0.7*np.array(cur_img_show, dtype=np.float32) + 0.3*np.array(heatmap, dtype=np.float32)
                    cur_img_with_heatmap = np.array(cur_img_with_heatmap, dtype=np.uint8)
                    new_im = Image.new('RGB', (msk_size * 4, msk_size))
                    new_im.paste(Image.fromarray(img_now.astype('uint8'), 'L'), (0, 0))
                    new_im.paste(Image.fromarray(cur_img_with_lbl.astype('uint8'), 'RGB'), (msk_size, 0))
                    new_im.paste(Image.fromarray(cur_img_with_pred.astype('uint8'), 'RGB'), (msk_size * 2, 0))
                    new_im.paste(Image.fromarray(cur_img_with_heatmap.astype('uint8'), 'RGB'), (msk_size * 3, 0))
                    new_im_name = names[ind] + "_%03d.jpg" % b
                    new_im_file = os.path.join(cur_show_path, new_im_name)
                    new_im.save(new_im_file)
                    # save asymmetry results
                    cur_sym_comp = sym_comp_t[ind, :, b, :, :].squeeze().data.cpu().numpy()
                    cur_sym_comp = cur_sym_comp * 255.0
                    cur_sym_comp = np.array(cur_sym_comp, dtype=np.uint8)

                    subject_asym_image = subject_asym_images_t[ind, :, b, :, :].squeeze().data.cpu().numpy()
                    subject_asym_image = subject_asym_image * 255.0
                    cur_subject_asym_image = np.array(subject_asym_image, dtype=np.uint8)

                    anatomy_asym_image = anatomy_asym_images_t[ind, :, b, :, :].squeeze().data.cpu().numpy()
                    anatomy_asym_image = anatomy_asym_image * 255.0
                    cur_anatomy_asym_image = np.array(anatomy_asym_image, dtype=np.uint8)

                    cur_subject_asym_conf = subject_asym_conf_t[ind, :, b, :, :].squeeze().data.cpu().numpy()
                    cur_subject_asym_conf[cur_subject_asym_conf<1e-5] = 0
                    cur_subject_asym_conf = map2fig(cur_subject_asym_conf)
                    cur_subject_asym_conf = np.array(cur_subject_asym_conf, dtype=np.uint8)
                    cur_anatomy_asym_conf = anatomy_asym_conf_t[ind, :, b, :, :].squeeze().data.cpu().numpy()
                    cur_anatomy_asym_conf = map2fig(cur_anatomy_asym_conf)
                    cur_anatomy_asym_conf = np.array(cur_anatomy_asym_conf, dtype=np.uint8)

                    cur_asym_map = asym_map_t[ind, :, b, :, :].squeeze().data.cpu().numpy()
                    cur_asym_map = map2fig(cur_asym_map)
                    cur_asym_map = np.array(cur_asym_map, dtype=np.uint8)

                    new_asym_img = Image.new('RGB', (msk_size * 3, msk_size * 3))
                    new_asym_img.paste(Image.fromarray(max_min_norm(img_now).astype('uint8'), 'L'), (0, 0))
                    new_asym_img.paste(Image.fromarray(max_min_norm(cur_img_with_lbl).astype('uint8'), 'RGB'),
                                       (msk_size, 0))
                    new_asym_img.paste(Image.fromarray(max_min_norm(cur_img_with_pred).astype('uint8'), 'RGB'),
                                       (msk_size * 2, 0))
                    new_asym_img.paste(Image.fromarray(max_min_norm(cur_sym_comp).astype('uint8'), 'L'),
                                       (0, msk_size))
                    new_asym_img.paste(Image.fromarray(max_min_norm(cur_subject_asym_image).astype('uint8'), 'L'),
                                       (msk_size, msk_size))
                    new_asym_img.paste(Image.fromarray(max_min_norm(cur_anatomy_asym_image).astype('uint8'), 'L'),
                                       (msk_size * 2, msk_size))
                    new_asym_img.paste(Image.fromarray(cur_asym_map.astype('uint8'), 'RGB'),
                                       (0, msk_size * 2))
                    new_asym_img.paste(Image.fromarray(cur_subject_asym_conf.astype('uint8'), 'RGB'),
                                       (msk_size, msk_size * 2))
                    new_asym_img.paste(Image.fromarray(cur_anatomy_asym_conf.astype('uint8'), "RGB"),
                                       (msk_size*2, msk_size * 2))
                    new_im_name = names[ind] + "_%03d.jpg" % b
                    new_im_file = os.path.join(cur_asym_path, new_im_name)
                    new_asym_img.save(new_im_file)

            batch_time.update(time.time() - end)
            end = time.time()

            if index % args.print_freq == 0:
                print('Test:[{0}/{1}]\t'
                      'Time {batch_time.val:.3f}({batch_time.avg:.3f})'.format(index, len(test_loader),
                                                                               batch_time=batch_time))
        print('The total test time is '+str(batch_time.sum))
        # print(MSK_PATH)
        print('making gif...')
        # make gif
        ct_name_list = os.listdir(ASYM_PATH)
        for ct_name in tqdm(ct_name_list):
            cur_asym_path = os.path.join(ASYM_PATH, ct_name)
            frame_name_list = os.listdir(cur_asym_path)
            frame_name_list.sort()
            frame_path_list = [os.path.join(cur_asym_path, x) for x in frame_name_list]
            frame_npy_list = [imageio.imread(x) for x in frame_path_list]
            gif_name = ct_name + ".gif"
            gif_path = os.path.join(GIF_PATH, gif_name)
            imageio.mimsave(gif_path, frame_npy_list)


def stn(x, theta):
    # theta must be (Bs, 3, 4) = [R|t]
    grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
    out = nn.functional.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
    return out


def max_min_norm(x):
    x_min = x.min()
    x_max = x.max()
    if x_max-x_min:
        return 255.0 * (x-x_min)/(x_max-x_min)
    else:
        return x


if __name__ == "__main__":
    main()

