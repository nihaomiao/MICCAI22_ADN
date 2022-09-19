# Train asymmetry extraction network D and segmentation network F
import argparse
import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from dataset import StrokeTrain3D
from model.unet3d.unet_model import ResidualUNet3D
from model.unet3d.losses import GeneralizedDiceLoss
from model.transform_net import PlaneFinder
import timeit
import math
from PIL import Image
import sys
from misc import Logger, map2fig
import torchvision.transforms as transforms

start = timeit.default_timer()
INPUT_SIZE = [256, 256]
BATCH_SIZE = 6
VAL_BATCH_SIZE = 6
NUM_SLICES = 40
NUM_EXAMPLES = 1000  # not real ct number in dataset because randomly choose CT per step
MAX_EPOCH = 100
USE_REG_EPOCH = 2  # increase this number to extend the warm-start epochs
GPU = "2, 3, 4, 5, 6, 7"
root_dir = "/data/hfn5052/StrokeCT/adn"
pos_weight = 1.0
gamma = 64./255
assert pos_weight == 1.0
useless_label = [4]
postfix = "-adn-wl%.1f" % (pos_weight)  # using dropout, sym rotation, gray-white matter
data_dir = "/data/StrokeCT/AISD_data_resample"
train_txt = "/data/StrokeCT/aisd_train.txt"
val_txt = "/data/StrokeCT/aisd_test.txt"
class_weight = np.array([1.0, pos_weight])
LEARNING_RATE = 1e-4
SEG_REC_RATE = 1
NUM_CLASSES = 2
POWER = 0.9
RANDOM_SEED = 1234
SEG_RESTORE_FROM = ""
ASYM_RESTORE_FROM = ""
# require pretrained gray-white matter and CSF segmentation model here
# we actually use SPM results to train a resunet3D segmentation model to ease the coding
GWM_SEG_RESTORE_FROM = "B0010_S003500.pth"
# require pretrained transformation network
ALIGN_RESTORE_FROM = "B0040_S012500.pth"
SEG_ASYM_RESTORE_FROM = ""
SNAPSHOT_DIR = osp.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = osp.join(root_dir, 'imgshots'+postfix)
WEIGHT_DECAY = 5e-4
LAMBDA_SEG = 1
LAMBDA_REG_1 = 10
LAMBDA_REG_2 = 10
NUM_EXAMPLES_PER_EPOCH = NUM_EXAMPLES
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
NUM_STEPS_USE_REG = NUM_STEPS_PER_EPOCH * USE_REG_EPOCH
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_PRED_EVERY = NUM_STEPS_PER_EPOCH * 5
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)
LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print("use dropout, probability is 0.2.")
print("useless label:", useless_label)
print("lr:", LEARNING_RATE)
print("gamma:", gamma)
print(postfix)
print(data_dir)
print("SEG, REG_1, REG_2:", LAMBDA_SEG, LAMBDA_REG_1, LAMBDA_REG_2)
print("num step to use regularization loss:", NUM_STEPS_USE_REG)
print("seg reduce rate:", SEG_REC_RATE)
print("save pred every:", SAVE_PRED_EVERY)


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


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UNet Network")
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--is-training", default=True,
                        help="Whether to freeze BN layers, False for Freezing")
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--num-workers", default=16)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=100, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Path to the text file listing the images in the dataset.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-jitter", default=True)
    parser.add_argument("--random-rotate", default=True)
    parser.add_argument("--skin-aug", default="-sa" in postfix)
    parser.add_argument("--inf", default="inf" in postfix)
    parser.add_argument("--random-scale", default="-sc" in postfix,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--seg-restore-from", type=str, default=SEG_RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--align-restore-from", type=str, default=ALIGN_RESTORE_FROM)
    parser.add_argument("--gwm-seg-restore-from", type=str, default=GWM_SEG_RESTORE_FROM)
    parser.add_argument("--seg-asym-restore-from", type=str, default=SEG_ASYM_RESTORE_FROM)
    parser.add_argument("--asym-restore-from", type=str, default=ASYM_RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label = label.cuda()
    BCELoss = nn.BCELoss()
    DiceLoss = GeneralizedDiceLoss(normalization="none")
    return BCELoss(pred, label), DiceLoss(pred.unsqueeze(dim=1), label.unsqueeze(dim=1))


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, actual_step):
    """Original Author: Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, actual_step * args.batch_size, MAX_ITER, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr*SEG_REC_RATE


def _voxel_accuracy(pred, target):
    accuracy_sum = 0.0
    for i in range(0, pred.shape[0]):
        out = pred[i] > 0.5
        accuracy = np.sum(out == target[i], dtype=np.float32) / out.size
        accuracy_sum += accuracy
    return accuracy_sum / pred.shape[0]


# change to ct dice
def _ct_accuracy(pred, target):
    dice_sum = 0.0
    for i in range(0, pred.shape[0]):
        out = pred[i] > 0.5
        overlap = np.sum(np.logical_and(out, target[i]))
        union = np.sum(out) + np.sum(target[i])
        if union:
            dice = 2*overlap/union
        else:
            dice = 1.0
        dice_sum += dice
    return dice_sum / pred.shape[0]


def main():
    """Create the model and start the training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    torch.manual_seed(args.random_seed)

    # tissue segmentation model
    gwm_seg_model = ResidualUNet3D(in_channels=1, out_channels=5, f_maps=32, final_sigmoid=False,
                                   use_transconv=False, use_dp=True, p=0.2, use_activation=False)
    gwm_seg_model = nn.DataParallel(gwm_seg_model)

    # alignment model
    align_model = PlaneFinder(is_train=False)
    # asymmetry separation network
    asym_model = ResidualUNet3D(in_channels=1, out_channels=1, f_maps=32, use_transconv=False, use_dp=True, p=0.2)
    # segmentation model
    seg_model = ResidualUNet3D(in_channels=1, out_channels=1, f_maps=32, use_transconv=False, use_dp=True, p=0.2)

    seg_model = nn.DataParallel(seg_model)
    asym_model = nn.DataParallel(asym_model)

    optimizer = optim.AdamW([{'params': asym_model.parameters(), 'lr': args.learning_rate},
                             {'params': seg_model.parameters(), 'lr': args.learning_rate*SEG_REC_RATE}],
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))

    if args.seg_restore_from:
        if os.path.isfile(args.seg_restore_from):
            print("=> loading checkpoint '{}'".format(args.seg_restore_from))
            checkpoint = torch.load(args.seg_restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            seg_model.load_state_dict(checkpoint['seg_state_dict'])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.seg_restore_from, args.start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.seg_restore_from))
            exit(-1)

    if args.align_restore_from:
        if os.path.isfile(args.align_restore_from):
            print("=> loading checkpoint '{}'".format(args.align_restore_from))
            checkpoint = torch.load(args.align_restore_from)
            align_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.align_restore_from, args.start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.align_restore_from))
            exit(-1)

    if args.gwm_seg_restore_from:
        if os.path.isfile(args.gwm_seg_restore_from):
            print("=> loading checkpoint '{}'".format(args.gwm_seg_restore_from))
            checkpoint = torch.load(args.gwm_seg_restore_from)
            gwm_seg_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.gwm_seg_restore_from, args.start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.gwm_seg_restore_from))
            exit(-1)

    if args.asym_restore_from:
        if os.path.isfile(args.asym_restore_from):
            print("=> loading checkpoint '{}'".format(args.asym_restore_from))
            checkpoint = torch.load(args.asym_restore_from)
            for name, _ in checkpoint['asym_state_dict'].items():
                try:
                    asym_model.state_dict()[name].copy_(checkpoint['asym_state_dict'][name])
                except:
                    print(name)
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.asym_restore_from, args.start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.asym_restore_from))
            exit(-1)

    if args.seg_asym_restore_from:
        if os.path.isfile(args.seg_asym_restore_from):
            print("=> loading checkpoint '{}'".format(args.seg_asym_restore_from))
            checkpoint = torch.load(args.seg_asym_restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            seg_model.load_state_dict(checkpoint["seg_state_dict"])
            asym_model.load_state_dict(checkpoint["asym_state_dict"])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.seg_asym_restore_from, args.start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.seg_asym_restore_from))
            exit(-1)

    align_model = nn.DataParallel(align_model)
    align_model.eval()
    align_model.cuda()

    gwm_seg_model.eval()
    gwm_seg_model.cuda()

    seg_model.train()
    seg_model.cuda()

    asym_model.train()
    asym_model.cuda()

    cudnn.benchmark = True
    trainloader = data.DataLoader(StrokeTrain3D(data_dir=data_dir, train_txt=train_txt,
                                                num_ct=NUM_EXAMPLES, gamma=gamma,
                                                is_mirror=args.random_mirror,
                                                is_jitter=args.random_jitter,
                                                is_rotate=args.random_rotate),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_losses = AverageMeter()
    reg_losses = AverageMeter()
    accuracy = AverageMeter()
    ct_accuracy = AverageMeter()

    print("class weight:", class_weight)

    cnt = 0
    actual_step = args.start_step
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)

            data_time.update(timeit.default_timer() - iter_end)

            images, labels, patch_name = batch

            images = images.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                images_t, images_r, images_t_f, _, M, M_inv = align_model(images)
                diff_t = images_t - images_t_f
                sym_comp_t = torch.zeros_like(images_t)
                sym_comp_t[diff_t > 0] = images_t[diff_t > 0]
                sym_comp_t[diff_t == 0] = images_t[diff_t == 0]
                sym_comp_t[diff_t < 0] = images_t_f[diff_t < 0]
                asym_map_t = nn.ReLU()(images_t_f - images_t)
                # infer gray-white matter
                gwm_logits_t = gwm_seg_model(images_t)
                gwm_msks_t = gwm_logits_t.argmax(dim=1)
                gm_msks_t = (gwm_msks_t == 1).type(torch.float32)
                gm_msks_t = gm_msks_t.unsqueeze(dim=1)
                gm_msks_t_f = transforms.functional.hflip(gm_msks_t)
                wm_msks_t = (gwm_msks_t == 2).type(torch.float32)
                wm_msks_t = wm_msks_t.unsqueeze(dim=1)
                gw_diff_t = torch.logical_and(gm_msks_t_f, wm_msks_t)
                # infer csf
                csf_msks_t = (gwm_msks_t == 3).type(torch.float32)
                csf_msks_t = csf_msks_t.unsqueeze(dim=1)

            labels_t = stn(labels.unsqueeze(dim=1), M[:, :3, :]).squeeze(dim=1)

            # deal with special value 4
            useless_area_t = torch.zeros_like(labels_t, dtype=torch.bool)
            for ll in useless_label:
                useless_area_t = useless_area_t + labels_t == ll
            labels_t[useless_area_t] = 0
            # consider value 1, 2, 3, 5 as infarct areas
            infarct_area_t = labels_t != 0
            labels_t[infarct_area_t] = 1

            # make gw_diff_t doesn't include stroke areas, i.e., all normal areas
            gw_diff_t = gw_diff_t * (labels_t == 0).unsqueeze(dim=1)
            csf_msks_t = csf_msks_t * (labels_t == 0).unsqueeze(dim=1)

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            # separate asym to be anatomical asym and other asym
            subject_asym_conf_t = asym_model(images_t)

            anatomy_asym_conf_t = asym_map_t - subject_asym_conf_t
            anatomy_asym_conf_t = nn.ReLU()(anatomy_asym_conf_t)
            subject_asym_images_t = images_t + anatomy_asym_conf_t
            pred_t = seg_model(subject_asym_images_t)
            pred_t = pred_t.squeeze(dim=1)

            bce_loss, dice_loss = loss_calc(pred_t, labels_t)
            seg_loss = bce_loss + dice_loss

            anatomy_asym_images_t = images_t + subject_asym_conf_t
            anatomy_asym_images_t = torch.clamp(anatomy_asym_images_t, max=sym_comp_t)

            if actual_step < NUM_STEPS_USE_REG:
                reg_bce_loss, reg_dice_loss = loss_calc(subject_asym_conf_t.squeeze(dim=1), labels_t)
                reg_loss = reg_bce_loss + reg_dice_loss
                LAMBDA_REG = LAMBDA_REG_1
            else:
                # 1. the size of subject asym should be the same as the size of stroke
                # 2. subject asym should from subject + anatomy
                # 3. anatomical asym should as large as possible
                # 4. subject asym should avoid gray-white matter difference
                # 5. subject asym should avoid csf
                subject_asym_msk_t = labels_t.unsqueeze(dim=1) == 1
                subject_asym_gt_t = asym_map_t * subject_asym_msk_t
                reg_loss1 = nn.L1Loss()(subject_asym_conf_t.mean(), subject_asym_gt_t.mean())
                sym_map_mask_t = asym_map_t == 0
                reg_loss2 = nn.L1Loss()(subject_asym_conf_t*sym_map_mask_t, torch.zeros_like(subject_asym_conf_t))
                reg_loss3 = -anatomy_asym_conf_t.mean()
                reg_loss4 = nn.L1Loss()(subject_asym_conf_t*gw_diff_t, torch.zeros_like(subject_asym_conf_t))
                reg_loss5 = nn.L1Loss()(subject_asym_conf_t*csf_msks_t, torch.zeros_like(subject_asym_conf_t))
                reg_loss = reg_loss1 + reg_loss2 + reg_loss3 + reg_loss4 + reg_loss5
                LAMBDA_REG = LAMBDA_REG_2

            loss = LAMBDA_SEG * seg_loss + LAMBDA_REG * reg_loss

            losses.update(loss.item(), pred_t.size(0))
            seg_losses.update(seg_loss.item(), pred_t.size(0))
            reg_losses.update(reg_loss.item(), pred_t.size(0))

            acc = _voxel_accuracy(pred_t.data.cpu().numpy(), labels_t.data.cpu().numpy())
            ct_acc = _ct_accuracy(pred_t.data.cpu().numpy(), labels_t.data.cpu().numpy())
            accuracy.update(acc, pred_t.size(0))
            ct_accuracy.update(ct_acc, pred_t.size(0))
            loss.backward()
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Seg {seg_loss.val:.4f} ({seg_loss.avg:.4f})\t'
                      'Reg {reg_loss.val:.4f} ({reg_loss.avg:.4f})\n'
                      'Pixel Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                      'CT Dice {ct_accuracy.val:.3f} ({ct_accuracy.avg:.3f})\t'
                      'Asym {asym:.4f}'.format(
                    cnt, actual_step, args.final_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, seg_loss=seg_losses,
                    reg_loss=reg_losses,
                    accuracy=accuracy, ct_accuracy=ct_accuracy, asym=subject_asym_conf_t.mean().item()))

            if actual_step % args.save_img_freq == 0:
                image_t = images_t.data.cpu().numpy()[0, 0, NUM_SLICES//2, :, :]
                image_sym_t = sym_comp_t.data.cpu().numpy()[0, 0, NUM_SLICES//2, :, :]
                msk_size = pred_t.size(-1)
                image_t = 255 * image_t
                image_sym_t = 255 * image_sym_t
                label_t = labels_t.data.cpu().numpy()[0, NUM_SLICES//2, :, :]
                label_t = 255 * (label_t != 0)
                single_pred_t = pred_t.data.cpu().numpy()[0, NUM_SLICES//2, :, :]
                single_pred_t = 255 * (single_pred_t > 0.5)
                # mix image with label
                cur_img_show = np.array(image_t, dtype=np.uint8)
                cur_img_show = np.repeat(np.expand_dims(cur_img_show, axis=2), 3, axis=2)
                cur_lbl_show = np.zeros((msk_size, msk_size, 3))
                cur_lbl_show[:, :, 0] = label_t
                cur_img_with_lbl = 0.7*np.array(cur_img_show, dtype=np.float32) + 0.3*np.array(cur_lbl_show, dtype=np.float32)
                cur_msk = np.repeat(np.expand_dims(label_t == 0, axis=2), 3, axis=2)
                cur_img_with_lbl[cur_msk] = cur_img_show[cur_msk]
                cur_img_with_lbl = np.array(cur_img_with_lbl, dtype=np.uint8)
                # mix image with pred
                cur_pred_show = np.zeros((msk_size, msk_size, 3))
                cur_pred_show[:, :, 0] = single_pred_t
                cur_img_with_pred = 0.7*np.array(cur_img_show, dtype=np.float32) + 0.3*np.array(cur_pred_show, dtype=np.float32)
                cur_pred_msk = np.repeat(np.expand_dims(single_pred_t == 0, axis=2), 3, axis=2)
                cur_img_with_pred[cur_pred_msk] = cur_img_show[cur_pred_msk]
                cur_img_with_pred = np.array(cur_img_with_pred, dtype=np.uint8)

                subject_asym_image = subject_asym_images_t.data.cpu().numpy()[0, 0, NUM_SLICES//2, :, :]
                subject_asym_image = 255.0 * subject_asym_image
                anatomy_asym_image = anatomy_asym_images_t.data.cpu().numpy()[0, 0, NUM_SLICES//2, :, :]
                anatomy_asym_image = 255.0 * anatomy_asym_image

                cur_subject_asym_conf = subject_asym_conf_t[0, 0, NUM_SLICES//2, :, :].squeeze().data.cpu().numpy()
                cur_subject_asym_conf = map2fig(cur_subject_asym_conf)
                cur_subject_asym_conf = np.array(cur_subject_asym_conf, dtype=np.uint8)
                cur_anatomy_asym_conf = anatomy_asym_conf_t[0, 0, NUM_SLICES//2, :, :].squeeze().data.cpu().numpy()
                cur_anatomy_asym_conf = map2fig(cur_anatomy_asym_conf)
                cur_anatomy_asym_conf = np.array(cur_anatomy_asym_conf, dtype=np.uint8)

                new_im = Image.new('RGB', (msk_size * 3, msk_size * 3))
                new_im.paste(Image.fromarray(image_t.astype('uint8'), 'L'), (0, 0))
                new_im.paste(Image.fromarray(cur_img_with_pred.astype('uint8'), 'RGB'), (msk_size, 0))
                new_im.paste(Image.fromarray(cur_img_with_lbl.astype('uint8'), 'RGB'), (msk_size * 2, 0))
                new_im.paste(Image.fromarray(subject_asym_image.astype('uint8'), 'L'), (0, msk_size))
                new_im.paste(Image.fromarray(anatomy_asym_image.astype('uint8'), 'L'), (msk_size, msk_size))
                new_im.paste(Image.fromarray(image_sym_t.astype('uint8'), 'L'), (msk_size * 2, msk_size))
                new_im.paste(Image.fromarray(cur_subject_asym_conf.astype('uint8'), 'RGB'), (0, msk_size * 2))
                new_im.paste(Image.fromarray(cur_anatomy_asym_conf.astype('uint8'), "RGB"), (msk_size, msk_size * 2))
                new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '_' + patch_name[0] + ".jpg"
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_pred_every == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'seg_state_dict': seg_model.state_dict(),
                            'asym_state_dict': asym_model.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            if actual_step >= args.final_step:
                break
            cnt += 1

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'seg_state_dict': seg_model.state_dict(),
                'asym_state_dict': asym_model.state_dict()},
               osp.join(args.snapshot_dir, 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


def bpDice(pd, gt, label):
    bp_pd = pd == label
    bp_gt = gt == label
    overlap = np.sum(np.logical_and(bp_pd, bp_gt))
    union = np.sum(bp_pd) + np.sum(bp_gt)
    if union:
        return 2*overlap/union
    else:
        return 1.0


def stn(x, theta):
    # theta must be (Bs, 3, 4) = [R|t]
    grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
    out = nn.functional.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
    return out


if __name__ == "__main__":
    main()
