import numpy as np
import torch
import logging
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device('cuda')
# 如果报错将to device都删掉

class MLJSMA(object):
    def __init__(self, model, dtypestr='float32', **kwargs):
        self.model = model
        self.model.to(device)
        # 如果报错将to device都删掉

    def generate_np(self, x, **kwargs):
        self.theta = kwargs['theta']
        self.gamma = kwargs['gamma']
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']

        x_shape = x.shape[1:]

        # print(x_shape)
        # (3, 448, 448)


        num_features = x_shape[0] * x_shape[1] * x_shape[2]
        num_labels = y_target.shape[1]
        # num_features = 602112
        # num_labels = 20

        iteration = 0
        sample = x.copy()

        x_t = torch.FloatTensor(x)
        sample_t = torch.FloatTensor(sample)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            x_t = x_t.cuda()
            sample_t = sample_t.cuda()

        output = self.model(x_t).cpu().detach().numpy()
        # 将输出改为1和0标识分类标签
        current = output.copy()
        current[current >= 0.5] = 1
        current[current < 0.5] = 0
        y_target = np.array(y_target)
        y_target[y_target == -1] = 0

        gradients, output = get_jacobian(self.model, sample_t, num_labels)
        # 计算雅可比矩阵
        gradients = np.asarray(gradients)
        # print(gradients.shape)-(20, 10, 3, 448, 448)-batch=10
        gradients = gradients.swapaxes(1, 0)
        # print(gradients.shape)-(10, 20, 3, 448, 448)
        original = current  # use original label as the reference
        logging.debug("start jsma attacking!")

        while np.any(current != y_target) and iteration < self.gamma:
            logging.info("%s out of %s has changed at iteration %s",
                         sum(np.any(current != original, axis=1)),
                         sample.shape[0],
                         iteration)
            mark = np.zeros(sample.shape[0])
            for idx in range(sample.shape[0]):
                if np.all(current[idx] == y_target[idx]):
                    logging.info("%s sample is succeed", idx)
                    continue
                if mark[idx] == 1:
                    logging.info("%s is no features to change")
                    continue
                y_differen = y_target[idx] - current[idx]
                jacobian = gradients[idx].reshape(num_labels, -1).T
                target_tmp = 0
                other_tmp = 0
                saliencymap = []

                for i in range(num_features):

                    for j in range(num_labels):
                        if y_differen[j] == 0:
                            other_tmp += jacobian[i][j]
                        else:
                            target_tmp += (jacobian[i][j] * y_differen[j])

                    if target_tmp < 0:
                        saliencymap.append(0)
                        target_tmp = 0
                        other_tmp = 0
                        continue
                    saliencymap.append(target_tmp-other_tmp)
                    target_tmp = 0
                    other_tmp = 0
                disturbe_num = 0
                print(math.floor(num_features/100))
                while disturbe_num < math.floor(num_features/100):
                    feature_max = np.argmax(saliencymap)
                    if saliencymap[feature_max] <= 0:
                        break
                    feature_max_idx1, feature_max_idx2, feature_max_idx3 = compute_idx(feature_max, x_shape)
                    sample[idx][feature_max_idx1][feature_max_idx2][feature_max_idx3] += self.theta
                    saliencymap[feature_max] = 0
                    disturbe_num += 1
                logging.info("%s sample %s features are changed", idx, disturbe_num)
                if disturbe_num == 0:
                    mark[idx] = 1

            sample = np.clip(sample, 0., 1.)
            sample_t = torch.FloatTensor(sample)
            if torch.cuda.is_available():
                sample_t = sample_t.cuda()
            gradients, output = get_jacobian(self.model, sample_t, num_labels)
            gradients = np.asarray(gradients)
            gradients = gradients.swapaxes(1, 0)
            current = output.copy()
            current[current >= 0.5] = 1
            current[current < 0.5] = 0
            iteration += 1

        return sample





# 计算影响最大像素点在原矩阵中的位置
def compute_idx(feature_max, x_shape):
    feature_max_idx1 = feature_max // (x_shape[1] * x_shape[2])
    feature_max_idx2 = (feature_max % (x_shape[1] * x_shape[2])) // x_shape[2]
    feature_max_idx3 = (feature_max % x_shape[2])
    return feature_max_idx1, feature_max_idx2, feature_max_idx3


def get_jacobian(model, x, noutputs):
    model.to(device)
    num_instaces = x.size()[0]
    # print(num_instaces) 10
    # print(noutputs) 20
    v = torch.eye(noutputs).cuda()
    # 对角线=1 的对角矩阵20*20
    jac = []

    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = True
    y = model(x)

    retain_graph = True
    for i in range(noutputs):
        if i == noutputs - 1:
            retain_graph = False

        y.backward(torch.unsqueeze(v[i], 0).repeat(num_instaces, 1), retain_graph=retain_graph)
        g = x.grad.cpu().detach().numpy()
        x.grad.zero_()
        jac.append(g)
    y = y.cpu().detach().numpy()
    return jac, y

