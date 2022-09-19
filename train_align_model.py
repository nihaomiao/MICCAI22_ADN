# Train alignment network, i.e., the Transformation network in our paper
import argparse
import torch
from torch.utils import data
import cv2
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from dataset import StrokeTrain3D
from model.transform_net import PlaneFinder
import timeit
import math
from PIL import Image
import sys
from misc import Logger

start = timeit.default_timer()
BATCH_SIZE = 40
NUM_SLICES = 40
NUM_EXAMPLES = 1000  # not real ct number in dataset because randomly choose CT per step
MAX_EPOCH = 500
IMG_SIZE = (256, 256)
GPU = "0"
root_dir = "/data/StrokeCT/align_net"
postfix = ""  # identifier for different experiments
data_dir = "/data/StrokeCT/AISD_data_resample"
train_txt = "/data/StrokeCT/aisd_train.txt"
LEARNING_RATE = 1e-5
NUM_CLASSES = 2
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = ""  # put pretrained model here
SNAPSHOT_DIR = osp.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = osp.join(root_dir, 'imgshots'+postfix)
WEIGHT_DECAY = 0.0005
NUM_EXAMPLES_PER_EPOCH = NUM_EXAMPLES
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_PRED_EVERY = NUM_STEPS_PER_EPOCH * MAX_EPOCH // 5
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)
LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(postfix)
print("lr:", LEARNING_RATE)


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
    parser = argparse.ArgumentParser(description="Transformation Network")
    parser.add_argument("--set-start", default=True)
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
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=100, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
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
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()


args = get_arguments()
print("mirror, jitter, rotate:", args.random_mirror, args.random_jitter, args.random_rotate)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, actual_step):
    """Original Author: Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, actual_step * args.batch_size, MAX_ITER, args.power)
    optimizer.param_groups[0]['lr'] = lr


def main():
    """Create the model and start the training."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.enabled = True
    torch.manual_seed(args.random_seed)

    model = PlaneFinder(is_train=True)
    print(model)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    if args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (step {})"
                  .format(args.restore_from, args.start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
            exit(-1)

    model.train()
    model.cuda()

    cudnn.benchmark = True
    trainloader = data.DataLoader(StrokeTrain3D(data_dir=data_dir, train_txt=train_txt,
                                                num_ct=NUM_EXAMPLES,
                                                is_mirror=args.random_mirror,
                                                is_jitter=args.random_jitter,
                                                is_rotate=args.random_rotate,
                                                vertical=True, resize=True),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_flip = AverageMeter()
    losses_rec = AverageMeter()

    cnt = 0
    actual_step = args.start_step
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)

            data_time.update(timeit.default_timer() - iter_end)

            images, _, patch_name = batch

            images = images.cuda()

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            images_t, images_r, _, view, _, _ = model(images)

            losses.update(model.loss_total.item(), images.size(0))
            losses_flip.update(model.loss_flip.item(), images.size(0))
            losses_rec.update(model.loss_rec.item(), images.size(0))

            model.loss_total.backward()
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            if actual_step % 100 == 0:
                show_view = view[0, :].data.cpu().numpy()
                rx = show_view[0] * model.x_rotation_range
                ry = show_view[1] * model.y_rotation_range
                rz = show_view[2] * model.z_rotation_range
                tx = show_view[3] * model.x_translation_range
                ty = show_view[4] * model.y_translation_range
                tz = show_view[5] * model.z_translation_range
                print('rx:%.2f ry:%.2f rz:%.2f tx:%.2f ty:%.2f tz:%.5f' % (rx, ry, rz, tx, ty, tz))

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                      'Flip {loss_flip.val:.5f} ({loss_flip.avg:.5f})\t'
                      'Rec {loss_rec.val:.5f} ({loss_rec.avg:.5f})'.format(
                    cnt, actual_step, args.final_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, loss_flip=losses_flip, loss_rec=losses_rec))

            if actual_step % args.save_img_freq == 0:
                msk_size = images.size(-1)
                image = images[0, 0, NUM_SLICES // 2, :, :]
                image = image.unsqueeze(-1).data.cpu().numpy()
                image = cv2.resize(255*image, (msk_size, msk_size), interpolation=cv2.INTER_NEAREST)
                image_t = images_t[0, 0, NUM_SLICES // 2, :, :]
                image_t = image_t.unsqueeze(-1).data.cpu().numpy()
                image_t = cv2.resize(255*image_t, (msk_size, msk_size), interpolation=cv2.INTER_NEAREST)
                image_r = images_r[0, 0, NUM_SLICES // 2, :, :]
                image_r = image_r.unsqueeze(-1).data.cpu().numpy()
                image_r = cv2.resize(255*image_r, (msk_size, msk_size), interpolation=cv2.INTER_NEAREST)

                new_im = Image.new('RGB', (msk_size * 3, msk_size))
                new_im.paste(Image.fromarray(image.astype('uint8'), 'L'), (0, 0))
                new_im.paste(Image.fromarray(image_t.astype('uint8'), 'L'), (msk_size, 0))
                new_im.paste(Image.fromarray(image_r.astype('uint8'), 'L'), (msk_size * 2, 0))
                new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '_' + patch_name[0] + ".jpg"
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_pred_every == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'state_dict': model.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            if actual_step >= args.final_step:
                break
            cnt += 1

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'state_dict': model.state_dict()},
               osp.join(args.snapshot_dir, 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == "__main__":
    main()
