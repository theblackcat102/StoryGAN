from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import PIL
import argparse
import os
import random
import sys
import pprint
import dateutil
import dateutil.tz
import numpy as np
import functools
import clevr_data as data
from clevr_data import video_transform
import pdb
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--model', dest='model',
                        help='directory which contain Model/ directory',
                        default='output/pororoSV_StoryGAN', type=str)

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfg/pororo.yml', type=str)

    parser.add_argument('--output', dest='output',
                        help='directory which contain Model/ directory',
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    num_gpu = len(cfg.GPU_ID.split(','))
    n_channels = 3

    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Resize( (cfg.IMSIZE, cfg.IMSIZE) ),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)
    testdataset = data.StoryDataset(dir_path, video_transforms, cfg.VIDEO_LEN, False)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=24,
        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))
    output_dir = './output/%s_%s/' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME)
    test_sample_save_dir = output_dir + 'test/'
    trainer = GANTrainer(output_dir, cfg, cfg.ST_WEIGHT, test_sample_save_dir, cfg.TENSORBOARD)
    trainer.evaluate(model_name, testloader, args.output)
