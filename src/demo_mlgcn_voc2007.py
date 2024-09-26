# @Time      :2019/12/15 16:16
# @Author    :zhounan
# @FileName  :attack_main_pytorch.py
import sys
import torch.multiprocessing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append('../')
import argparse
import torch
import numpy as np
import logging
from tqdm import tqdm
import torchvision.transforms as transforms
from model.ML_GCN_model.models import gcn_resnet101_attack
from data.data_voc import Voc2007Classification
from model.ML_GCN_model.util import Warp
from data.data_voc import write_object_labels_csv
from src.attack_model import AttackModel
from torch import nn
from utils.until import *

parser = argparse.ArgumentParser(description='multi-label attack')
parser.add_argument('--data', default='../data/VOC2007', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N', help='batch size (default: 32)')
parser.add_argument('--adv_batch_size', default=10, type=int,
                    metavar='N', help='batch size 32')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--adv_method', default='PGD', type=str, metavar='N',
                    help='attack method: ml_cw, ml_deepfool, mla_lp, FGM, MI-FGSM, BIM, PGD, FGM, SDA')
parser.add_argument('--target_type', default='show_all', type=str, metavar='N',
                    help='target method: hide_all, show_all')
parser.add_argument('--adv_file_path', default='../data/VOC2007/classification_mlgcn_adv.csv', type=str, metavar='N',
                    help='all image names and their labels ready to attack')
parser.add_argument('--adv_save_x', default='../adv_save/mlgcn/VOC2007/', type=str, metavar='N',
                    help='save adversiral examples')
parser.add_argument('--adv_begin_step', default=0, type=int, metavar='N',
                    help='which step to start attacking according to the batch size')
parser.add_argument('--eval', default=1, type=int, metavar='N',
                    help='0 is attack, 1 is evaluate adv ')


class NewModel(nn.Module):
    def __init__(self, model):
        super(NewModel, self).__init__()
        self.model = model
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, input):
        o = torch.sigmoid(self.model(input))
        return o


def get_target_label(y, target_type):
    '''
    :param y: numpy, y in {0, 1}
    :param A: list, label index that we want to reverse
    :param C: list, label index that we don't care
    :return:
    '''
    y = y.copy()
    # o to -1
    y[y == 0] = -1
    if target_type == 'hide_all':
        y[y == 1] = -1
    if target_type == 'show_all':
        y[y == -1] = 1
    return y


def gen_adv_file(model, target_type, adv_file_path):
    tqdm.monitor_interval = 0
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    test_dataset = Voc2007Classification(args.data, 'val', inp_name='../data/VOC2007/voc_glove_word2vec.pkl')
    test_dataset.transform = data_transforms
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    output = []
    image_name_list = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            x = input[0]
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
            image_name_list.extend(list(input[1]))
        output = np.asarray(output)
        y = np.asarray(y)
        image_name_list = np.asarray(image_name_list)

    # choose x which can be well classified and contains two or more label to prepare attack
    pred = (output >= 0.5) + 0
    y[y == -1] = 0
    true_idx = []
    for i in range(len(pred)):
        if (y[i] == pred[i]).all() and np.sum(y[i]) >= 1:
            true_idx.append(i)
    adv_image_name_list = image_name_list[true_idx]
    adv_y = y[true_idx]
    y = y[true_idx]

    y_target = get_target_label(adv_y, target_type)
    y_target[y_target == 0] = -1
    y[y == 0] = -1

    y_target = y_target[0:1000]
    y = y[0:1000]
    adv_image_name_list = adv_image_name_list[0:1000]

    print(len(adv_image_name_list))
    adv_labeled_data = {}
    for i in range(len(adv_image_name_list)):
        adv_labeled_data[adv_image_name_list[i]] = y[i]
    write_object_labels_csv(adv_file_path, adv_labeled_data)

    # save target y and ground-truth y to prepare attack
    # value is {-1,1}
    np.save('../adv_save/mlgcn/VOC2007/y_target.npy', y_target)
    np.save('../adv_save/mlgcn/VOC2007/y.npy', y)


def evaluate_model(model):
    model.eval()
    tqdm.monitor_interval = 0
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                     std=model.image_normalization_std)
    test_data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = Voc2007Classification(args.data, 'test', filename='test')
    test_dataset.transform = test_data_transforms
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    output = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            x = input[0]
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())

        output = np.asarray(output)
        y = np.asarray(y)

    pred = (output >= 0.5) + 0
    y[y == -1] = 0

    from utils import evaluate_metrics
    metric = evaluate_metrics.evaluate(y, output, pred)
    print(metric)


def evaluate_adv(state):
    model = state['model']
    y_target = state['y_target']
    y = state['y']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = [file for file in os.listdir(adv_folder_path) if file.startswith(state['target_type'])]
    adv_file_list.sort(key=lambda x: int(x[13:-4]))
    adv_file_list = adv_file_list[: 20]
    adv = []

    # if args.adv_method == 'mla_lp':
    #     end_epoch = 5
    # else:
    #     end_epoch = 100

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))
        # if i >= end_epoch:
        #     break

    adv = np.asarray(adv)
    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor()
    ])
    adv_dataset = Voc2007Classification(args.data, 'mlgcn_adv', inp_name='../data/VOC2007/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms

    adv_dataset.images = adv_dataset.images[0:len(adv)]
    y_target = y_target[0:len(adv)]
    dl2 = torch.utils.data.DataLoader(adv_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)
    dl2 = tqdm(dl2, desc='ADV')

    adv_output = []
    norm_1 = []
    norm = []
    max_r = []
    mean_r = []
    rmsd = []
    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x[0][0].cpu().numpy()

            batch_r = (batch_adv_x - batch_test_x)
            batch_r_255 = ((batch_adv_x) * 255) - ((batch_test_x) * 255)
            batch_norm = [np.linalg.norm(r.flatten()) for r in batch_r]
            batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
            norm.extend(batch_norm)
            rmsd.extend(batch_rmsd)
            norm_1.extend(np.sum(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
    adv_output = np.asarray(adv_output)
    adv_pred = adv_output.copy()
    adv_pred[adv_pred >= (0.5 + 0)] = 1
    adv_pred[adv_pred < (0.5 + 0)] = -1
    print(adv_pred.shape)
    print(y_target.shape)

    max_r_limit_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.5, 1]
    mean_r_limit_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.5, 1]

    for i, max_r_limit in enumerate(max_r_limit_list):
        norm_t = np.asarray(norm)
        max_r_t = np.asarray(max_r)
        mean_r_t = np.asarray(mean_r)
        rmsd_t = np.asarray(rmsd)
        norm_1_t = np.asarray(norm_1)

        adv_pred_match_target = np.all((adv_pred == y_target), axis=1) + 0
        adv_pred_match_target = adv_pred_match_target * ((max_r_t <= max_r_limit) + 0)
        attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten().tolist()
        np.save('../attack_result/mlgcn_voc2007_{}_max_r_attack_fail_idx_{}.npy'.format(args.adv_method, max_r_limit),
                attack_fail_idx)

        norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
        max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
        norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
        mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
        rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)

        metrics = dict()
        adv_pred[adv_pred == -1] = 0
        y[y == -1] = 0
        metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
        metrics['norm'] = np.mean(norm_t)
        metrics['norm_1'] = np.mean(norm_1_t)
        metrics['rmsd'] = np.mean(rmsd_t)
        metrics['max_r'] = np.mean(max_r_t)
        metrics['mean_r'] = np.mean(mean_r_t)
        metrics['mean_pos_label'] = np.sum(adv_pred) / len(adv_pred)
        metrics['mean_add_label'] = np.sum(adv_pred - y) / len(adv_pred)
        print()
        logging.info(str(metrics))
        # write_to_excel(metrics, base_row=1, args=args, sheet=i, filename='max_r_result.xls')

    for i, mean_r_limit in enumerate(mean_r_limit_list):
        norm_t = np.asarray(norm)
        max_r_t = np.asarray(max_r)
        mean_r_t = np.asarray(mean_r)
        rmsd_t = np.asarray(rmsd)
        norm_1_t = np.asarray(norm_1)

        adv_pred_match_target = np.all((adv_pred == y_target), axis=1) + 0
        adv_pred_match_target = adv_pred_match_target * ((mean_r_t <= mean_r_limit) + 0)
        attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten().tolist()
        np.save('../attack_result/mlgcn_voc2007_{}_mean_r_attack_fail_idx_{}.npy'.format(args.adv_method, mean_r_limit),
                attack_fail_idx)

        norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
        max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
        norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
        mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
        rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)

        metrics = dict()
        metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
        metrics['norm'] = np.mean(norm_t)
        metrics['norm_1'] = np.mean(norm_1_t)
        metrics['rmsd'] = np.mean(rmsd_t)
        metrics['max_r'] = np.mean(max_r_t)
        metrics['mean_r'] = np.mean(mean_r_t)
        print()
        logging.info(str(metrics))
        # write_to_excel(metrics, base_row=1, args=args, sheet=i, filename='mean_r_result.xls')


def evaluate_adv_many(state):
    model = state['model']
    y_target = state['y_target']
    y = state['y']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = [file for file in os.listdir(adv_folder_path) if file.startswith(state['target_type'])]
    adv_file_list.sort(key=lambda x: int(x[13:-4]))
    adv = []

    # if args.adv_method == 'mla_lp':
    #     end_epoch = 5
    # else:
    #     end_epoch = 100

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))
        # if i >= end_epoch:
        #     break

    adv = np.asarray(adv)
    dl1 = torch.utils.data.DataLoader(adv[0: 200],
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor()
    ])
    adv_dataset = Voc2007Classification(args.data, 'mlgcn_adv', inp_name='../data/VOC2007/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms

    adv_dataset.images = adv_dataset.images[0:len(adv)]
    y_target = y_target[0:len(adv)]
    dl2 = torch.utils.data.DataLoader(adv_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)
    dl2 = tqdm(dl2, desc='ADV')

    adv_output = []
    norm_1 = []
    norm = []
    max_r = []
    mean_r = []
    rmsd = []
    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x[0][0].cpu().numpy()

            batch_r = (batch_adv_x - batch_test_x)
            batch_r_255 = ((batch_adv_x) * 255) - ((batch_test_x) * 255)
            batch_norm = [np.linalg.norm(r.flatten()) for r in batch_r]
            batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
            norm.extend(batch_norm)
            rmsd.extend(batch_rmsd)
            norm_1.extend(np.sum(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
    adv_output = np.asarray(adv_output)
    adv_pred = adv_output.copy()
    adv_pred[adv_pred >= (0.5 + 0)] = 1
    adv_pred[adv_pred < (0.5 + 0)] = 0
    y[y == -1] = 0
    mean_label = np.sum(y) / len(y)
    print("mean label={}".format(mean_label))

    add_of_means = [mean_label + 1, mean_label + 2, mean_label + 4, mean_label + 8, mean_label + 16, mean_label + 32]

    for i, add_of_mean in enumerate(add_of_means):
        norm_t = np.asarray(norm)
        max_r_t = np.asarray(max_r)
        mean_r_t = np.asarray(mean_r)
        rmsd_t = np.asarray(rmsd)
        norm_1_t = np.asarray(norm_1)

        if add_of_mean >= 20:
            add_of_mean = 20

        adv_pred_match_target = (np.sum(adv_pred, axis=1) >= int(add_of_mean)) + 0
        attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten().tolist()
        np.save('../attack_result/mlgcn_voc2007_{}_mean_r_attack_fail_idx_{}.npy'.format(args.adv_method, add_of_mean),
                attack_fail_idx)

        norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
        max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
        norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
        mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
        rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)

        metrics = dict()
        metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
        metrics['norm'] = np.mean(norm_t)
        metrics['norm_1'] = np.mean(norm_1_t)
        metrics['rmsd'] = np.mean(rmsd_t)
        metrics['max_r'] = np.mean(max_r_t)
        metrics['mean_r'] = np.mean(mean_r_t)
        print()
        logging.info(str(metrics))


def display_adv(state):
    from PIL import Image
    import matplotlib.pyplot as plt

    # 40 constains 3,14 labels
    display_idx = 40

    # 51 constanins
    display_idx = 51

    model = state['model']
    y_target = state['y_target']
    y = state['y']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[13:-4]))
    adv = []

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    adv = np.asarray(adv)
    print(adv.shape)
    adv = adv[display_idx:display_idx + 1]

    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor()
    ])
    adv_dataset = Voc2007Classification(args.data, 'mlgcn_adv', inp_name='../data/VOC2007/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms

    # adv_dataset.images = adv_dataset.images[0:len(adv)]
    adv_dataset.images = adv_dataset.images[display_idx:display_idx + 1]
    # y_target = y_target[0:len(adv)]
    y = y[display_idx:display_idx + 1]
    dl2 = torch.utils.data.DataLoader(adv_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)
    dl2 = tqdm(dl2, desc='ADV')

    with torch.no_grad():

        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
                batch_test_x = batch_test_x[0][0]
                batch_test_x = batch_test_x.cuda()

            p1 = model(batch_adv_x).cpu().numpy()
            p2 = model(batch_test_x).cpu().numpy()

            bar = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            color = ['tab:blue' for i in range(20)]
            color[3] = 'tab:red'
            color[-6] = 'tab:red'

            plt.figure(figsize=(4, 4))
            plt.barh(bar, p1[0], color=color)
            plt.vlines(0.5, 0, 20, colors="r", linestyles="dashed")
            plt.xlim(0, 1)
            plt.xticks([0.0, 0.5, 1.0])
            plt.savefig('../attack_result/mlgcn_voc2007_{}_adv_conf_{}.pdf'.format(args.adv_method, display_idx),
                        format='pdf', bbox_inches='tight')
            plt.clf()

            plt.figure(figsize=(4, 4))
            plt.barh(bar, p2[0], color=color)
            plt.vlines(0.5, 0, 20, colors="r", linestyles="dashed")
            plt.xlim(0, 1)
            plt.xticks([0.0, 0.5, 1.0])
            plt.savefig('../attack_result/mlgcn_voc2007_test_conf_{}.pdf'.format(args.adv_method, display_idx),
                        format='pdf', bbox_inches='tight')
            plt.clf()

            batch_adv_x = batch_adv_x.cpu().numpy()

            temp = np.uint8(batch_adv_x[0] * 255)
            temp = np.transpose(temp, [1, 2, 0])
            img = Image.fromarray(temp)
            img.save('../attack_result/mlgcn_voc2007_{}_adv_{}.jpg'.format(args.adv_method, display_idx))

            batch_test_x = batch_test_x.cpu().numpy()
            temp = np.uint8(batch_test_x[0] * 255)
            temp = np.transpose(temp, [1, 2, 0])
            img = Image.fromarray(temp)
            img.save('../attack_result/mlgcn_voc2007_test_{}.jpg'.format(args.adv_method, display_idx))


def display_adv_with_eps(state):
    from PIL import Image
    import matplotlib.pyplot as plt

    # 51 constanins 1, 15 labels
    display_idx = 51

    model = state['model']
    y_target = state['y_target']
    y = state['y']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x: int(x[13:-4]))
    adv = []

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    adv = np.asarray(adv)
    print(adv.shape)
    adv = adv[display_idx:display_idx + 1]

    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor()
    ])
    adv_dataset = Voc2007Classification(args.data, 'mlgcn_adv', inp_name='../data/VOC2007/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms

    # adv_dataset.images = adv_dataset.images[0:len(adv)]
    adv_dataset.images = adv_dataset.images[display_idx:display_idx + 1]
    # y_target = y_target[0:len(adv)]
    y = y[display_idx:display_idx + 1]
    dl2 = torch.utils.data.DataLoader(adv_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)
    dl2 = tqdm(dl2, desc='ADV')

    with torch.no_grad():

        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
                batch_test_x = batch_test_x[0][0]
                batch_test_x = batch_test_x.cuda()

            p1 = model(batch_adv_x).cpu().numpy()
            p2 = model(batch_test_x).cpu().numpy()

            bar = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            color = ['tab:blue' for i in range(20)]
            color[1] = 'tab:red'
            color[-6] = 'tab:red'

            plt.figure(figsize=(4, 4))
            plt.barh(bar, p1[0], color=color)
            plt.vlines(0.5, 0, 20, colors="r", linestyles="dashed")
            plt.xlim(0, 1)
            plt.xticks([0.0, 0.5, 1.0])
            plt.savefig('../attack_result/mlgcn_voc2007_{}_adv_conf_{}.pdf'.format(args.adv_method, display_idx),
                        format='pdf', bbox_inches='tight')
            plt.clf()

            # plt.figure(figsize=(4, 4))
            # plt.barh(bar, p2[0], color=color)
            # plt.vlines(0.5, 0, 20, colors="r", linestyles="dashed")
            # plt.xlim(0, 1)
            # plt.xticks([0.0, 0.5, 1.0])
            # plt.savefig('../attack_result/mlgcn_voc2007_test_conf_{}.pdf'.format(args.adv_method, display_idx),
            #             format='pdf', bbox_inches='tight')
            # plt.clf()

            batch_adv_x = batch_adv_x.cpu().numpy()

            temp = np.uint8(batch_adv_x[0] * 255)
            temp = np.transpose(temp, [1, 2, 0])
            img = Image.fromarray(temp)
            img.save('../attack_result/mlgcn_voc2007_{}_adv_{}.jpg'.format(args.adv_method, display_idx))

            batch_test_x = batch_test_x.cpu().numpy()
            temp = np.uint8(batch_test_x[0] * 255)
            temp = np.transpose(temp, [1, 2, 0])
            img = Image.fromarray(temp)
            img.save('../attack_result/mlgcn_voc2007_test_{}.jpg'.format(display_idx))

            eps = batch_adv_x[0] - batch_test_x[0]
            temp = np.uint8(eps * 255)
            temp = np.transpose(temp, [1, 2, 0])
            img = Image.fromarray(temp)
            img.save('../attack_result/mlgcn_voc2007_{}_eps_{}.jpg'.format(args.adv_method, display_idx))


def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    print(args)
    use_gpu = torch.cuda.is_available()

    # set seed
    torch.manual_seed(123)
    if use_gpu:
        torch.cuda.manual_seed_all(123)
    np.random.seed(123)

    init_log(os.path.join(args.adv_save_x, args.adv_method, args.target_type + '.log'))

    # define dataset
    num_classes = 20

    # load torch model
    model = gcn_resnet101_attack(num_classes=num_classes,
                                 t=0.4,
                                 adj_file='../data/VOC2007/voc_adj.pkl',
                                 word_vec_file='../data/VOC2007/voc_glove_word2vec.pkl',
                                 save_model_path='../checkpoint/mlgcn/VOC2007/model_best.pth.tar')
    model = NewModel(model)
    model.eval()
    if use_gpu:
        model = model.cuda()

    gen_adv_file(model, args.target_type, args.adv_file_path)

    # transfor image to torch tensor
    # the tensor size is [chnnel, height, width]
    # the tensor value in [0,1]
    global normalize
    normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                     std=model.image_normalization_std)
    # test_data_transforms = transforms.Compose([
    #     Warp(args.image_size),
    #     transforms.ToTensor(),
    #     normalize
    # ])

    test_data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor()
    ])
    adv_dataset = Voc2007Classification(args.data, 'mlgcn_adv', inp_name='../data/VOC2007/voc_glove_word2vec.pkl')
    adv_dataset.transform = test_data_transforms
    adv_loader = torch.utils.data.DataLoader(adv_dataset,
                                             batch_size=args.adv_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)

    # load target y and ground-truth y
    # value is {-1,1}
    y_target = np.load('../adv_save/mlgcn/VOC2007/y_target.npy')
    y = np.load('../adv_save/mlgcn/VOC2007/y.npy')

    state = {'model': model,
             'data_loader': adv_loader,
             'adv_method': args.adv_method,
             'target_type': args.target_type,
             'adv_batch_size': args.adv_batch_size,
             'y_target': y_target,
             'y': y,
             'adv_save_x': os.path.join(args.adv_save_x, args.adv_method, args.target_type + '.npy'),
             'adv_begin_step': args.adv_begin_step,
             'model_type': 'mlgcn'
             }

    if args.eval == 0:
        # start attack
        attack_model = AttackModel(state)
        attack_model.attack()
    elif args.eval == 1:
        evaluate_adv(state)
    elif args.eval == 2:
        evaluate_adv_many(state)
    elif args.eval == 3:
        evaluate_model(model)
    elif args.eval == 4:
        display_adv_with_eps(state)
    # evaluate_model(model)


if __name__ == '__main__':
    main()