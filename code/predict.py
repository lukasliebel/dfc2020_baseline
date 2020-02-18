import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from datasets import SEN12MS, DFC2020, TIFFDir
from models.deeplab import DeepLab
from models.unet import UNet
from utils import labels_to_dfc
from utils import mycmap as dfc_cmap
from utils import mypatches as dfc_legend


# define and parse arguments
parser = argparse.ArgumentParser()

# config
parser.add_argument('--config_file', type=str, default="args.pkl",
                    help='path to config file (default: ./args.conf)')
parser.add_argument('--checkpoint_file', type=str, default="checkpoint.pth",
                    help='path to checkpoint file (default: ./checkpoint.pth)')

# general
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for prediction (default: 32)')
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers for dataloading (default: 4)')
parser.add_argument('--score', action='store_true', default=False,
                    help='score prediction results using ground-truth data')

# data
parser.add_argument('--dataset', type=str, default="sen12ms_holdout",
                    choices=['sen12ms_holdout', 'dfc2020_val', 'dfc2020_test',
                             'tiff_dir'],
                    help='type of dataset (default: sen12ms_holdout)')
parser.add_argument('--data_dir', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--out_dir', type=str, default="results",
                    help='path to output dir (default: ./results)')
parser.add_argument('--preview_dir', type=str, default=None,
                    help='path to preview dir (default: no previews)')

args = parser.parse_args()
print("="*20, "PREDICTION CONFIG", "="*20)
for arg in vars(args):
    print('{0:20}  {1}'.format(arg, getattr(args, arg)))
print()

# load config
train_args = pkl.load(open(args.config_file, "rb"))
print("="*20, "TRAIN CONFIG", "="*20)
for arg in vars(train_args):
    print('{0:20}  {1}'.format(arg, getattr(train_args, arg)))
print()

# create output dir
os.makedirs(args.out_dir, exist_ok=True)

# create preview dir
if args.preview_dir is not None:
    os.makedirs(args.preview_dir, exist_ok=True)

# set flags for GPU processing if available
if torch.cuda.is_available():
    args.use_gpu = True
    if torch.cuda.device_count() > 1:
        raise NotImplementedError("multi-gpu prediction not implemented! "
                                  + "try to run script as: "
                                  + "CUDA_VISIBLE_DEVICES=0 predict.py")
else:
    args.use_gpu = False

# load dataset
if args.dataset == "sen12ms_holdout":
    dataset = SEN12MS(args.data_dir,
                      subset="holdout",
                      no_savanna=train_args.no_savanna,
                      use_s2hr=train_args.use_s2hr,
                      use_s2mr=train_args.use_s2mr,
                      use_s2lr=train_args.use_s2lr,
                      use_s1=train_args.use_s1)
    gt_id = "lc"
elif args.dataset == "tiff_dir":
    assert not args.score
    dataset = TIFFDir(args.data_dir,
                      no_savanna=train_args.no_savanna,
                      use_s2hr=train_args.use_s2hr,
                      use_s2mr=train_args.use_s2mr,
                      use_s2lr=train_args.use_s2lr,
                      use_s1=train_args.use_s1)
    gt_id = "pred"
else:
    dfc2020_subset = args.dataset.split("_")[-1]
    dataset = DFC2020(args.data_dir,
                      subset=dfc2020_subset,
                      no_savanna=train_args.no_savanna,
                      use_s2hr=train_args.use_s2hr,
                      use_s2mr=train_args.use_s2mr,
                      use_s2lr=train_args.use_s2lr,
                      use_s1=train_args.use_s1)
    gt_id = "dfc"
n_classes = dataset.n_classes
n_inputs = dataset.n_inputs

# set up dataloader
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.workers,
                        pin_memory=True,
                        drop_last=False)

# set up network
if train_args.model == "deeplab":
    model = DeepLab(num_classes=n_classes,
                    backbone='resnet',
                    pretrained_backbone=False,
                    output_stride=train_args.out_stride,
                    sync_bn=False,
                    freeze_bn=False,
                    n_in=n_inputs)
else:
    model = UNet(n_classes=n_classes,
                 n_channels=n_inputs)
if args.use_gpu:
    model = model.cuda()

# restore network weights
state = torch.load(args.checkpoint_file)
step = state["step"]
model.load_state_dict(state["model_state_dict"])
model.eval()
print("loaded checkpoint from step", step)

# initialize scoring if ground-truth is available
if args.score:
    import metrics
    conf_mat = metrics.ConfMatrix(n_classes)

# predict samples
n = 0
for batch in tqdm(dataloader, desc="[Pred]"):

    # unpack sample
    image = batch['image']
    if args.score:
        target = batch['label']

    # move data to gpu if model is on gpu
    if args.use_gpu:
        image = image.cuda()
        if args.score:
            target = target.cuda()

    # forward pass
    with torch.no_grad():
        prediction = model(image)

    # convert to 256x256 numpy arrays
    prediction = prediction.cpu().numpy()
    prediction = np.argmax(prediction, axis=1)
    if args.score:
        target = target.cpu().numpy()

    # save predictions
    for i in range(prediction.shape[0]):

        n += 1
        id = batch["id"][i].replace("_s2_", "_" + gt_id + "_")

        output = labels_to_dfc(prediction[i, :, :], train_args.no_savanna)

        output = output.astype(np.uint8)
        output_img = Image.fromarray(output)
        output_img.save(os.path.join(args.out_dir, id))

        # update error metrics
        if args.score:
            gt = labels_to_dfc(target[i, :, :], train_args.no_savanna)
            conf_mat.add(target[i, :, :], prediction[i, :, :])

        # save preview
        if args.preview_dir is not None:

            # colorize labels
            cmap = dfc_cmap()
            output = (output - 1) / 10
            output = cmap(output)[:, :, 0:3]
            if args.score:
                gt = (gt - 1) / 10
                gt = cmap(gt)[:, :, 0:3]
            display_channels = dataset.display_channels
            brightness_factor = dataset.brightness_factor

            if args.score:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2)
            img = image.cpu().numpy()[i, display_channels, :, :]
            img = np.rollaxis(img, 0, 3)
            s1 = image.cpu().numpy()[i, -2:-1, :, :]
            s1 = np.rollaxis(s1, 0, 3)
            ax1.imshow(np.clip(img * brightness_factor, 0, 1))
            ax1.set_title("input")
            ax1.axis("off")
            ax2.imshow(output)
            ax2.set_title("prediction")
            ax2.axis("off")
            if args.score:
                ax3.imshow(gt)
                ax3.set_title("label")
                ax3.axis("off")
            lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
                             handles=dfc_legend(), ncol=2, title="DFC Classes")
            ttl = fig.suptitle(id, y=0.75)
            plt.savefig(os.path.join(args.preview_dir, id),
                        bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
            plt.close()

# print scoring results
if args.score:
    print("AA\t", conf_mat.get_aa())
    print("mIoU\t", conf_mat.get_mIoU())
