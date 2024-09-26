import sys

sys.path.append('../')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import torchvision.models as models
import torch
import torch.optim as optim
from model.ml_liw_model.models import Inceptionv3Rank
from data.data_voc import *
from data.data_coco import *
import torchvision.transforms as transforms
import shutil
from data.data_voc import *
from data.data_nuswide import *

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', default='../../data/VOC2007', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def instance_wise_loss(output, y):
    y_i = torch.eq(y, torch.ones_like(y))
    y_not_i = torch.eq(y, -torch.ones_like(y))

    column = torch.unsqueeze(y_i, 2)
    row = torch.unsqueeze(y_not_i, 1)
    truth_matrix = column * row
    column = torch.unsqueeze(output, 2)
    row = torch.unsqueeze(output, 1)
    sub_matrix = column - row
    exp_matrix = torch.exp(-sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, (1, 2))
    y_i_sizes = torch.sum(y_i, 1)
    y_i_bar_sizes = torch.sum(y_not_i, 1)
    normalizers = y_i_sizes * y_i_bar_sizes
    normalizers_zero = torch.logical_not(torch.eq(normalizers, torch.zeros_like(normalizers)))
    normalizers = normalizers[normalizers_zero]
    sums = sums[normalizers_zero]
    loss = sums / normalizers
    loss = torch.sum(loss)
    return loss


def label_wise_loss(output, y):
    output = torch.transpose(output, 0, 1)
    y = torch.transpose(y, 0, 1)
    return instance_wise_loss(output, y)


def criterion(output, y):
    loss = 0.5 * instance_wise_loss(output, y) + label_wise_loss(output, y)
    return loss


def save_checkpoint(model, is_best, best_score, save_model_path, filename='checkpoint.pth.tar'):
    filename_ = filename
    filename = os.path.join(save_model_path, filename_)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print('save model {filename}'.format(filename=filename))
    torch.save(model.state_dict(), filename)
    if is_best:
        filename_best = 'model_best.pth.tar'
        filename_best = os.path.join(save_model_path, filename_best)
        shutil.copyfile(filename, filename_best)

        filename_best = os.path.join(save_model_path, 'model_best_{score:.4f}.pth.tar'.format(score=best_score))
        shutil.copyfile(filename, filename_best)


def train(model, epoch, optimizer, train_loader):
    total_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data[0].cuda(), target.cuda()
        else:
            data = data[0].cuda()
        optimizer.zero_grad()
        output = model(data)
        # print(data.size())
        # print(target.size())
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))


def test(model, test_loader):
    from utils import evaluate_metrics
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data, target = data[0].cuda(), target.cuda()
            else:
                data = data[0].cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            outputs.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    targets[targets == -1] = 0
    pred = outputs.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    metrics = evaluate_metrics.evaluate(targets, outputs, pred, model='mlliw')
    print(metrics)
    return test_loss


def main_voc2007():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)
    train_dataset = Voc2007Classification(args.data, 'train')
    val_dataset = Voc2007Classification(args.data, 'val')
    test_dataset = Voc2007Classification(args.data, 'test', filename='test')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.transform = data_transforms
    val_dataset.transform = data_transforms
    test_dataset.transform = data_transforms
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    num_classes = 20
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.aux_logit = False
    for param in model.parameters():
        param.requires_grad = False

    model = Inceptionv3Rank(model, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.model.fc.parameters())
    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader)
        val_loss = test(model, val_loader)
        scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            test(model, test_loader)
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../checkpoint/mlliw/voc2007/',
                        filename='voc2007_checkpoint.pth.tar')


def main_voc2012():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)

    train_dataset = Voc2012Classification(args.data, 'train', inp_name='../../data/VOC2012/voc_glove_word2vec.pkl')
    val_dataset = Voc2012Classification(args.data, 'val', inp_name='../../data/VOC2012/voc_glove_word2vec.pkl')
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    train_dataset.transform = data_transforms
    val_dataset.transform = data_transforms

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)
    num_classes = 20
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.aux_logit = False
    for param in model.parameters():
        param.requires_grad = False

    model = Inceptionv3Rank(model, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.model.fc.parameters())
    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader)
        val_loss = test(model, val_loader)
        scheduler.step(epoch)

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../../checkpoint/mlliw/VOC2012/',
                        filename='voc2012_checkpoint.pth.tar')


def main_nuswide():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)

    train_dataset = NusWide(args.data, 'train', inp_name='../../data/NUSWIDE/glove_word2vec.pkl')
    val_dataset = NusWide(args.data, 'val', inp_name='../../data/NUSWIDE/glove_word2vec.pkl')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ])
    val_transform = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
        normalize, ])
    train_dataset.transform = train_transform
    # train_dataset.target_transform = self._state('train_target_transform')
    val_dataset.transform = val_transform
    # val_dataset.target_transform = self._state('val_target_transform')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)

    num_classes = 81
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.aux_logit = False
    for param in model.parameters():
        param.requires_grad = False

    model = Inceptionv3Rank(model, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.model.fc.parameters())
    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader)
        val_loss = test(model, val_loader)
        scheduler.step(epoch)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../../checkpoint/mlliw/nuswide/',
                        filename='nuswide_checkpoint.pth.tar')


def main_coco():
    torch.manual_seed(123)
    if use_cuda:
        torch.cuda.manual_seed_all(123)

    train_dataset = COCO2014(args.data, 'train', inp_name='../../data/COCO/coco_glove_word2vec.pkl')
    val_dataset = COCO2014(args.data, 'val', inp_name='../../data/COCO/coco_glove_word2vec.pkl')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize, ])
    val_transform = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
        normalize, ])
    train_dataset.transform = train_transform
    # train_dataset.target_transform = self._state('train_target_transform')
    val_dataset.transform = val_transform
    # val_dataset.target_transform = self._state('val_target_transform')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)

    num_classes = 80
    model = models.inception_v3(pretrained=True)
    model.eval()
    model.aux_logit = False
    for param in model.parameters():
        param.requires_grad = False

    model = Inceptionv3Rank(model, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.model.fc.parameters())
    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    best_loss = 1e5
    # Train
    for epoch in range(1, args.epochs + 1):
        train(model, epoch, optimizer, train_loader)
        val_loss = test(model, val_loader)
        scheduler.step(epoch)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint(model, is_best, best_loss,
                        save_model_path='../../checkpoint/mlliw/coco/',
                        filename='coco_checkpoint.pth.tar')


if __name__ == '__main__':
    main_voc2007()
