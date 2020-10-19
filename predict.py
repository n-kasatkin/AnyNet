import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import save_image
import time
from dataloader import loader_husky as DataLoader_
import utils.logger as logger
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import cv2
import numpy as np

import models.anynet

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')


# Modified by NK
parser.add_argument('--left_images', type=str, default=None)
parser.add_argument('--right_images', type=str, default=None)
parser.add_argument('--save_dir', type=str, default='/home/kasatkin/Projects/AnyNet/output/husky_ouster_64/')
parser.add_argument('--print_avg_time', action='store_true', help='avg prediction time')

args = parser.parse_args()


def main():
    global args
    os.makedirs(args.save_dir, exist_ok=True)

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("=> no pretrained model found at '{}'")


    left_images = [os.path.join(args.left_images, file) for file in sorted(os.listdir(args.left_images))]
    right_images = [os.path.join(args.right_images, file) for file in sorted(os.listdir(args.right_images))]
    loader = DataLoader_.myImageFloder(left_images, right_images)

    cudnn.benchmark = True
    test(loader, model)


def test(dataloader, model):
    stages = 3 + args.with_spn
    model.eval()
    times = []
    for idx, (imgL, imgR) in tqdm(enumerate(dataloader)):
        imgL = imgL.float().cuda().unsqueeze(0)
        imgR = imgR.float().cuda().unsqueeze(0)

        with torch.no_grad():
            t = time.time()
            outputs = model(imgL, imgR)
            times.append(time.time() - t)
            if args.print_avg_time and idx%50 == 0:
                print(np.mean(times))
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)

            filename = str(idx).rjust(8, '0') + '.png'
            save_path = os.path.join(args.save_dir, filename)
            save_depth(output, save_path)


def save_depth(tensor, path):
    img_cpu = np.clip(np.asarray(tensor.cpu()), 0, 2**16)
    img_save = (img_cpu * 256.0).astype(np.uint16)
    # print(img_save.shape)
    # time.sleep(100)
    cv2.imwrite(path, img_save[0, :, :])


if __name__ == '__main__':
    main()
