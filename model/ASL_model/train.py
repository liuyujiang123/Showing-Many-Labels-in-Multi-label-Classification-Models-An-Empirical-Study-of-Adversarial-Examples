#@Time      :2021/1/14 15:26
#@Author    :zhounan
#@FileName  :train.py.py
import sys
sys.path.append('../../')

import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os

from model.ASL_model.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from model.ASL_model.models import create_model
import numpy as np
from model.ASL_model.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized

from data.data_voc import Voc2012Classification
import shutil
from tqdm import tqdm
from model.ASL_model import OneCycle, cutout
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='../../data/VOC2012', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--model-name', default='tresnet_xl')
parser.add_argument('--model-path', default='../../checkpoint/asl/VOC2012/PASCAL_VOC_TResNet_xl.pth', type=str)
parser.add_argument('--num-classes', default=20)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')


def main_voc2012():
    args = parser.parse_args()
    args.batch_size = args.batch_size

    # setup model
    print('creating the model...')

    # VOC2007 as a pretrain model for VOC2012
    #state = torch.load('../../checkpoint/asl/VOC2007/PASCAL_VOC_TResNet_xl_448_96.0.pth', map_location='cpu')
    #args.num_classes = state['num_classes']
    args.num_classes = 20
    model = create_model(args).cuda()
    #model.load_state_dict(state['model'], strict=True)
    print('done\n')

    # Data loading code
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        # 0.5 factor
        #cutout.Cutout(n_holes=1, length=16),
    ])

    val_data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = Voc2012Classification(args.data, 'train')
    dataset.transform = data_transforms
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    val_dataset = Voc2012Classification(args.data, 'val')
    val_dataset.transform = val_data_transforms
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=args.workers)

    loss_func = AsymmetricLossOptimized()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)

    epoch = 25
    all_steps = math.ceil(dataset.__len__() / args.batch_size) * epoch
    scheduler = OneCycle.OneCycle(optimizer, nb=all_steps, max_lr=0.0002)
    scheduler.step()

    best_mAP = 0
    for i in range(epoch):
        print('epoch: {}/{}'.format(i, epoch))
        model.train()
        for input, target in tqdm(data_loader, desc='train', ncols=60):
            input[0] = input[0].cuda()
            target = target.cuda()
            output = model(input[0])
            target[target==-1] = 0
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

        model.eval()
        val_mAP = validate_multi(val_loader, model, args)
        is_best = val_mAP > best_mAP
        if is_best:
            best_mAP = val_mAP
        print('best mAP:{}'.format(best_mAP))
        save_checkpoint(model, is_best, best_mAP,
                        save_model_path='../../checkpoint/asl/VOC2012',
                        filename='PASCAL_VOC_TResNet_xl.pth')

def save_checkpoint(model, is_best, best_score, save_model_path, filename):
    filename = os.path.join(save_model_path, filename)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    torch.save(model.state_dict(), filename)

    if is_best:
        filename_best = 'model_best.pth.tar'
        filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)
        filename_best = os.path.join(save_model_path, 'model_best_{score:.4f}.pth.tar'.format(score=best_score))
        shutil.copyfile(filename, filename_best)

def validate_multi(val_loader, model, args):
    print("starting actuall validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    mAP_meter = AverageMeter()

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        input[0] = input[0].cuda()
        input = input[0]
        target[target == -1] = 0
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(args.thre).long()
        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                prec=prec, rec=rec))
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("mAP score:", mAP_score)

    return mAP_score

if __name__ == '__main__':
    main_voc2012()