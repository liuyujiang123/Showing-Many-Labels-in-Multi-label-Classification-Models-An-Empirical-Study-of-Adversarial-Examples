#@Time      :2019/12/15 16:16
#@Author    :zhounan
#@FileName  :attack_main_pytorch.py
import sys
sys.path.append('../')

import argparse
import torch
import os
import numpy as np
import logging
from tqdm import tqdm
import torchvision.transforms as transforms
from data.data_nuswide import NusWide
from src.attack_model import AttackModel
from model.ASL_model.models import create_model
from PIL import Image
from torch import nn
import csv
from utils.until import *

parser = argparse.ArgumentParser(description='multi-label attack')
parser.add_argument('--data', default='../data/NUSWIDE', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model_path', default='../checkpoint/asl/NUSWIDE/NUS_WIDE_TRresNet_L_448_65.2.pth', type=str)
parser.add_argument('--num_classes', default=81)
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')

parser.add_argument('--adv_batch_size', default=24, type=int,
                    metavar='N', help='batch size 32')
parser.add_argument('--adv_method', default='ml_deepfool', type=str, metavar='N',
                    help='attack method: ml_cw, ml_deepfool, mla_lp')
parser.add_argument('--target_type', default='hide_all', type=str, metavar='N',
                    help='target method: hide_all')
parser.add_argument('--adv_file_path', default='../data/NUSWIDE/nus_wide_data_asl_adv.csv', type=str, metavar='N',
                    help='all image names and their labels ready to attack')
parser.add_argument('--adv_save_x', default='../adv_save/asl/NUSWIDE/', type=str, metavar='N',
                    help='save adversiral examples')
parser.add_argument('--adv_begin_step', default=0, type=int, metavar='N',
                    help='which step to start attacking according to the batch size')
parser.add_argument('--eval', default=1, type=int, metavar='N',
                    help='0 is attack, 1 is evaluate adv ')

class NewModel(nn.Module):
    def __init__(self, model):
        super(NewModel, self).__init__()
        self.model = model
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
    return y

def gen_adv_file(model, target_type, adv_file_path):
    tqdm.monitor_interval = 0
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = NusWide(args.data, phase='val')
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

    pred = (output >= 0.5) + 0
    y[y==-1] = 0
    true_idx = []
    for i in range(len(pred)):
        if (y[i] == pred[i]).all() and np.sum(y[i]) >= 1:
            true_idx.append(i)
    adv_image_name_list = image_name_list[true_idx]
    adv_y = y[true_idx]
    y = y[true_idx]
    y_target = get_target_label(adv_y, target_type)
    y_target[y_target==0] = -1
    y[y==0] = -1

    new_image = [test_dataset.img_name_list[i] for i in true_idx]
    new_tag = [test_dataset.tag_list[i] for i in true_idx]
    y = y[0:1000]
    y_target = y_target[0:1000]
    new_image = new_image[0:1000]
    new_tag = new_tag[0:1000]


    # img_list_path = os.path.join('../data/NUSWIDE/', '{}ImagelistFilter.txt'.format('asl_adv'))
    # tag_list_path = os.path.join('../data/NUSWIDE/', '{}_Tags81Filter.txt'.format('asl_adv'))

    with open(adv_file_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filepath', 'label' , 'split_name'])

        for img, tag in zip(new_image, new_tag):
            tags = []
            for i, t in enumerate(tag):
                if t == 1:
                    tags.append(test_dataset.tags[i])
            row_info = ['images/' + img, str(tags), 'asl_adv']
            writer.writerow(row_info)
    np.save('../adv_save/asl/NUSWIDE/y_target.npy', y_target)
    np.save('../adv_save/asl/NUSWIDE/y.npy', y)


def evaluate_model(model):
    model.eval()
    tqdm.monitor_interval = 0
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = NusWide(args.data, phase='val')
    test_dataset.transform = data_transforms
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    output = []
    y = []
    test_loader = tqdm(test_loader, desc='val')
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
    metric = evaluate_metrics.evaluate(y, output, pred, model='asl')
    print(metric)

def evaluate_adv(state):
    model = state['model']
    y_target = state['y_target']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x:int(x[13:-4]))
    adv = []
    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    adv = np.asarray(adv)
    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    adv_dataset = NusWide(args.data, phase='asl_adv')
    adv_dataset.transform = data_transforms

    adv_dataset.img_name_list = adv_dataset.img_name_list[0:len(adv)]
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
    adv_pred[adv_pred >= (0.5+0)] = 1
    adv_pred[adv_pred < (0.5+0)] = -1
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
        np.save('../attack_result/asl_nuswide_{}_max_r_attack_fail_idx_{}.npy'.format(args.adv_method, max_r_limit), attack_fail_idx)

        norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
        max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
        norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
        mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
        rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)

        metrics = dict()
        #y_target[y_target == -1] = 0
        metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
        metrics['norm'] = np.mean(norm_t)
        metrics['norm_1'] = np.mean(norm_1_t)
        metrics['rmsd'] = np.mean(rmsd_t)
        metrics['max_r'] = np.mean(max_r_t)
        metrics['mean_r'] = np.mean(mean_r_t)
        print()
        logging.info(str(metrics))
        write_to_excel(metrics, base_row=31, args=args, sheet=i, filename='max_r_result.xls')

    for i, mean_r_limit in enumerate(mean_r_limit_list):
        norm_t = np.asarray(norm)
        max_r_t = np.asarray(max_r)
        mean_r_t = np.asarray(mean_r)
        rmsd_t = np.asarray(rmsd)
        norm_1_t = np.asarray(norm_1)

        adv_pred_match_target = np.all((adv_pred == y_target), axis=1) + 0
        adv_pred_match_target = adv_pred_match_target * ((mean_r_t <= mean_r_limit) + 0)
        attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten().tolist()
        np.save('../attack_result/asl_nuswide_{}_mean_r_attack_fail_idx_{}.npy'.format(args.adv_method, mean_r_limit),
                attack_fail_idx)

        norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
        max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
        norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
        mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
        rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)

        metrics = dict()
        #y_target[y_target == -1] = 0
        metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
        metrics['norm'] = np.mean(norm_t)
        metrics['norm_1'] = np.mean(norm_1_t)
        metrics['rmsd'] = np.mean(rmsd_t)
        metrics['max_r'] = np.mean(max_r_t)
        metrics['mean_r'] = np.mean(mean_r_t)
        print()
        logging.info(str(metrics))
        write_to_excel(metrics, base_row=31, args=args, sheet=i, filename='mean_r_result.xls')

def evaluate_defense_adv(state):
    from advertorch import defenses
    jpeg_denfense = defenses.JPEGFilter()

    model = state['model']
    y_target = state['y_target']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x:int(x[13:-4]))
    adv = []

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    adv = np.asarray(adv)
    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor()
    ])
    adv_dataset = NusWide(args.data, phase='mlgcn_adv', inp_name='../data/NUSWIDE/glove_word2vec.pkl')
    adv_dataset.transform = data_transforms

    adv_dataset.img_name_list = adv_dataset.img_name_list[0:len(adv)]
    y_target = y_target[0:len(adv)]
    dl2 = torch.utils.data.DataLoader(adv_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    dl2 = tqdm(dl2, desc='ADV')

    adv_output = []
    norm = []
    norm_1 = []
    max_r = []
    mean_r = []
    rmsd = []
    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
            batch_adv_x = jpeg_denfense(batch_adv_x)

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

    max_r_limit_list = [1]

    for i, max_r_limit in enumerate(max_r_limit_list):
        norm_t = np.asarray(norm)
        max_r_t = np.asarray(max_r)
        mean_r_t = np.asarray(mean_r)
        rmsd_t = np.asarray(rmsd)
        norm_1_t = np.asarray(norm_1)

        adv_pred_match_target = np.all((adv_pred == y_target), axis=1) + 0
        adv_pred_match_target = adv_pred_match_target * ((max_r_t <= max_r_limit) + 0)
        attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten().tolist()

        norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
        max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
        norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
        mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
        rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)

        metrics = dict()
        # y_target[y_target == -1] = 0
        metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
        metrics['norm'] = np.mean(norm_t)
        metrics['norm_1'] = np.mean(norm_1_t)
        metrics['rmsd'] = np.mean(rmsd_t)
        metrics['max_r'] = np.mean(max_r_t)
        metrics['mean_r'] = np.mean(mean_r_t)
        logging.info(str(metrics))
        write_to_excel(metrics, base_row=31, args=args, sheet=0, filename='defense_result.xls')



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


    # load torch model
    logging.info('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)

    model = NewModel(model)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    logging.info('done\n')


    if not os.path.exists(args.adv_file_path):
        gen_adv_file(model, args.target_type, args.adv_file_path)

    # transfor image to torch tensor
    # the tensor size is [chnnel, height, width]
    # the tensor value in [0,1]
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    adv_dataset = NusWide(args.data, phase='asl_adv')
    adv_dataset.transform = data_transforms
    test_loader = torch.utils.data.DataLoader(adv_dataset,
                                              batch_size=args.adv_batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)

    # load target y and ground-truth y
    # value is {-1,1}
    y_target = np.load('../adv_save/asl/NUSWIDE/y_target.npy')
    y = np.load('../adv_save/asl/NUSWIDE/y.npy')

    state = {'model': model,
             'data_loader': test_loader,
             'adv_method': args.adv_method,
             'target_type': args.target_type,
             'adv_batch_size': args.adv_batch_size,
             'y_target':y_target,
             'y': y,
             'adv_save_x': os.path.join(args.adv_save_x, args.adv_method, args.target_type + '.npy'),
             'adv_begin_step': args.adv_begin_step,
             'model_type': 'asl'
             }

    if args.eval == 0:
        # start attack
        attack_model = AttackModel(state)
        attack_model.attack()
    elif args.eval == 1:
        evaluate_adv(state)
    #evaluate_model(model)

if __name__ == '__main__':
    main()