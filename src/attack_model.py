# @Time      :2019/12/15 18:55
# @Author    :zhounan
# @FileName  :attack_model.py
from attacks.ml_cw import MLCarliniWagnerL2
from attacks.ml_rank1 import MLRank1
from attacks.ml_rank2 import MLRank2
from attacks.ml_deepfool import MLDeepFool
from attacks.mla_lp import MLLP
import numpy as np
import math
import os
from tqdm import tqdm
import logging
from advertorch import attacks
import torch
from torch import nn

tqdm.monitor_interval = 0


class AttackModel():
    def __init__(self, state):
        self.state = state
        self.y_target = state['y_target']
        self.y = state['y']
        self.data_loader = tqdm(state['data_loader'], desc='ADV', ncols=60)
        self.model = state['model']
        self.adv_save_x = state['adv_save_x']
        self.adv_batch_size = state['adv_batch_size']
        self.adv_begin_step = state['adv_begin_step']
        self.attack_model = None

    def attack(self):
        clip_min = 0.
        clip_max = 1.

        if self.state['adv_method'] == 'mla_lp':
            self.attack_model = MLLP(self.model)
            params = {'y_target': None,
                      'max_iter': 10,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.mla_lp(params)
        elif self.state['adv_method'] == 'ml_cw':
            self.attack_model = MLCarliniWagnerL2(self.model)
            params = {'binary_search_steps': 10,
                      'y_target': None,
                      'max_iterations': 1000,
                      'learning_rate': 0.01,
                      'batch_size': self.adv_batch_size,
                      'initial_const': 1e5,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.ml_cw(params)
        elif self.state['adv_method'] == 'ml_rank1':
            self.attack_model = MLRank1(self.model)
            params = {'binary_search_steps': 10,
                      'y_target': None,
                      'max_iterations': 1000,
                      'learning_rate': 0.01,
                      'batch_size': self.adv_batch_size,
                      'initial_const': 1e5,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.ml_rank1(params)
        elif self.state['adv_method'] == 'ml_rank2':
            self.attack_model = MLRank2(self.model)
            params = {'binary_search_steps': 10,
                      'y_target': None,
                      'max_iterations': 1000,
                      'learning_rate': 0.01,
                      'batch_size': self.adv_batch_size,
                      'initial_const': 1e5,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.ml_rank2(params)
        elif self.state['adv_method'] == 'ml_deepfool':
            self.attack_model = MLDeepFool(self.model)
            params = {'y_target': None,
                      'max_iter': 20,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.ml_deepfool(params)

        elif self.state['adv_method'] == 'FGSM':
            params = {'y_target': None,
                      'eps': 0.3,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.fgsm(params)
        elif self.state['adv_method'] == 'MI-FGSM':
            params = {'y_target': None,
                      'eps': 0.3,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            self.mi_fgsm(params)
        elif self.state['adv_method'] == 'FGM':
            params = {'y_target': self.y_target,
                      'eps': 1.0,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            # default: 0.03
            self.FGM(params)
        elif self.state['adv_method'] == 'BIM':
            params = {'y_target': self.y_target,
                      'eps': 0.3,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            # default: 0.1 - 10 - 0.05
            self.BIM(params)
        elif self.state['adv_method'] == 'PGD':
            params = {'y_target': self.y_target,
                      'eps': 0.3,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            # default: 0.3 - 40 - 0.01
            self.PGD(params)
        elif self.state['adv_method'] == 'SDA':
            params = {'y_target': self.y_target,
                      'eps': 10000.0,
                      'eps_iter': 300.0,
                      'clip_min': clip_min,
                      'clip_max': clip_max}
            # default:0.3-40-0.01
            self.SDA(params)

    def mla_lp(self, params):
        _, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)
        A = A_pos + A_neg
        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)
        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            params['y_target'] = self.y_target[begin:end]

            adv = self.attack_model.generate_np(input[0].cpu().numpy(), A[begin:end], **params)
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def ml_cw(self, params):
        _, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            params['y_target'] = self.y_target[begin:end]

            adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def ml_rank1(self, params):
        y_tor, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            params['y_target'] = self.y_target[begin:end]
            params['y_tor'] = y_tor[begin:end]
            adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def ml_rank2(self, params):
        y_tor, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)

        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            params['y_target'] = self.y_target[begin:end]
            params['y_tor'] = y_tor[begin:end]
            params['A_pos'] = A_pos[begin:end]
            params['A_neg'] = A_neg[begin:end]
            params['B_pos'] = B_pos[begin:end]
            params['B_neg'] = B_neg[begin:end]

            adv = self.attack_model.generate_np(input[0].cpu().numpy(), **params)
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def ml_deepfool(self, params):
        _, A_pos, A_neg, B_pos, B_neg = get_target_set(self.y, self.y_target)
        A = A_pos + A_neg

        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            params['y_target'] = self.y_target[begin:end]

            adv = self.attack_model.generate_np(input[0].cpu().numpy(), A[begin:end], **params)
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

        # adv_list = []
        #
        # for i in range(begin_step, step):
        #     tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
        #     tmp_file = np.load(tmp_file_path)
        #     adv_list.extend(tmp_file)
        # np.save(self.adv_save_x, np.asarray(adv_list))

    def fgsm(self, params):
        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        class MLLoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, o, y):
                o = torch.clamp(o, 1e-6, 1 - 1e-6)
                loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
                loss = loss.sum()
                return loss

        adversary = attacks.GradientSignAttack(self.model,
                                               loss_fn=MLLoss(),
                                               eps=params['eps'],
                                               clip_min=params['clip_min'],
                                               clip_max=params['clip_max'],
                                               targeted=True)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            x = input[0].cpu().numpy()
            y_target = self.y_target[begin:end]
            y_target[y_target == -1] = 0

            x_t = torch.FloatTensor(x)
            y_target_t = torch.FloatTensor(y_target)
            if torch.cuda.is_available():
                x_t = x_t.cuda()
                y_target_t = y_target_t.cuda()
            x_t.requires_grad = True

            adv = adversary.perturb(x_t, y_target_t)
            adv = adv.detach().cpu().numpy()
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

    def FGM(self, params):
        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        class MLLoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, o, y):
                o = torch.clamp(o, 1e-6, 1 - 1e-6)
                loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
                loss = loss.sum()
                return loss

        adversary = attacks.GradientAttack(self.model,
                                           loss_fn=MLLoss(),
                                           eps=1.,
                                           clip_min=params['clip_min'],
                                           clip_max=params['clip_max'],
                                           targeted=True)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            x = input[0].cpu().numpy()
            print(x.shape)
            y_target = self.y_target[begin:end]
            y_target[y_target == -1] = 0

            x_t = torch.FloatTensor(x)
            y_target_t = torch.FloatTensor(y_target)
            if torch.cuda.is_available():
                x_t = x_t.cuda()
                y_target_t = y_target_t.cuda()
            x_t.requires_grad = True
            adv = adversary.perturb(x_t, y_target_t)
            adv = adv.detach().cpu().numpy()
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

    def mi_fgsm(self, params):
        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        class MLLoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, o, y):
                o = torch.clamp(o, 1e-6, 1 - 1e-6)
                loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
                loss = loss.sum()
                return loss

        adversary = attacks.MomentumIterativeAttack(self.model,
                                                    loss_fn=MLLoss(),
                                                    eps=params['eps'],
                                                    clip_min=params['clip_min'],
                                                    clip_max=params['clip_max'],
                                                    targeted=True)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            x = input[0].cpu().numpy()
            y_target = self.y_target[begin:end]
            y_target[y_target == -1] = 0

            x_t = torch.FloatTensor(x)
            y_target_t = torch.FloatTensor(y_target)
            if torch.cuda.is_available():
                x_t = x_t.cuda()
                y_target_t = y_target_t.cuda()
            x_t.requires_grad = True

            adv = adversary.perturb(x_t, y_target_t)
            adv = adv.detach().cpu().numpy()
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

    def BIM(self, params):
        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        class MLLoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, o, y):
                o = torch.clamp(o, 1e-6, 1 - 1e-6)
                loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
                loss = loss.sum()
                return loss

        adversary = attacks.LinfBasicIterativeAttack(self.model,
                                                     loss_fn=MLLoss(),
                                                     eps=params['eps'],
                                                     nb_iter=30,
                                                     eps_iter=0.02,
                                                     clip_min=params['clip_min'],
                                                     clip_max=params['clip_max'],
                                                     targeted=True)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            x = input[0].cpu().numpy()
            y_target = self.y_target[begin:end]
            y_target[y_target == -1] = 0

            x_t = torch.FloatTensor(x)
            y_target_t = torch.FloatTensor(y_target)
            if torch.cuda.is_available():
                x_t = x_t.cuda()
                y_target_t = y_target_t.cuda()
            x_t.requires_grad = True

            adv = adversary.perturb(x_t, y_target_t)
            adv = adv.detach().cpu().numpy()

            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

    def PGD(self, params):
        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        class MLLoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, o, y):
                o = torch.clamp(o, 1e-6, 1 - 1e-6)
                loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
                loss = loss.sum()
                return loss

        adversary = attacks.PGDAttack(self.model,
                                      loss_fn=MLLoss(),
                                      eps=params['eps'],
                                      nb_iter=30,
                                      eps_iter=0.02,
                                      clip_min=params['clip_min'],
                                      clip_max=params['clip_max'],
                                      targeted=True)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            x = input[0].cpu().numpy()
            y_target = self.y_target[begin:end]
            y_target[y_target == -1] = 0

            x_t = torch.FloatTensor(x)
            y_target_t = torch.FloatTensor(y_target)
            if torch.cuda.is_available():
                x_t = x_t.cuda()
                y_target_t = y_target_t.cuda()
            x_t.requires_grad = True

            adv = adversary.perturb(x_t, y_target_t)
            adv = adv.detach().cpu().numpy()
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)

    def SDA(self, params):
        tmp_folder_path = os.path.join(os.path.dirname(self.adv_save_x), 'tmp/')
        new_folder(tmp_folder_path)
        begin_step = self.adv_begin_step
        batch_size = self.adv_batch_size
        step = math.ceil(len(self.y_target) / batch_size)
        logging.info(params)

        class MLLoss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, o, y):
                o = torch.clamp(o, 1e-6, 1 - 1e-6)
                loss = -(y * torch.log(o) + (1 - y) * torch.log(1 - o))
                loss = loss.sum()
                return loss

        adversary = attacks.SparseL1DescentAttack(self.model,
                                                  loss_fn=MLLoss(),
                                                  eps=params['eps'],
                                                  nb_iter=40,
                                                  eps_iter=params['eps_iter'],
                                                  clip_min=params['clip_min'],
                                                  clip_max=params['clip_max'],
                                                  targeted=True)

        for i, (input, target) in enumerate(self.data_loader):
            logging.info('batch is {}'.format(i))
            logging.info('{} generator data, length is {}'.format(i, len(input[0])))
            if i < begin_step:
                continue
            params['batch_size'] = len(target)
            begin = i * batch_size
            end = begin + len(target)
            x = input[0].cpu().numpy()
            y_target = self.y_target[begin:end]
            y_target[y_target == -1] = 0

            x_t = torch.FloatTensor(x)
            y_target_t = torch.FloatTensor(y_target)
            if torch.cuda.is_available():
                x_t = x_t.cuda()
                y_target_t = y_target_t.cuda()
            x_t.requires_grad = True

            adv = adversary.perturb(x_t, y_target_t)
            adv = adv.detach().cpu().numpy()
            tmp_file_path = os.path.join(tmp_folder_path, os.path.basename(self.adv_save_x) + '_' + str(i) + '.npy')
            np.save(tmp_file_path, adv)


def new_folder(file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_target_set(y, y_target):
    y[y == 0] = -1
    A_pos = np.logical_and(np.not_equal(y, y_target), y == 1) + 0
    A_neg = np.logical_and(np.not_equal(y, y_target), y == -1) + 0
    B_pos = np.logical_and(np.equal(y, y_target), y == 1) + 0
    B_neg = np.logical_and(np.equal(y, y_target), y == -1) + 0

    y_tor = A_pos * -2 + -1 * B_neg + 1 * B_pos + 2 * A_neg
    return y_tor, A_pos, A_neg, B_pos, B_neg
