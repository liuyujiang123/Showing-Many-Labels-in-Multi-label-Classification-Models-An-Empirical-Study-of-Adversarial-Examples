import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import sys
sys.path.append('../../')

import argparse
from model.ML_GCN_model.engine import *
from model.ML_GCN_model.models import *
from data.data_voc import *
from data.data_nuswide import *
from data.data_coco import *
import pickle

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', default='../../data/VOC2012', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[40], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,

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
parser.add_argument('-n', '--nodes', default=1,
                    type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')

def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()


    train_dataset = Voc2007Classification(args.data, 'train', inp_name='../../data/VOC2007/voc_glove_word2vec.pkl')
    val_dataset = Voc2007Classification(args.data, 'val', inp_name='../../data/VOC2007/voc_glove_word2vec.pkl')

    num_classes = 20

    # load model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='../../data/VOC2007/voc_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = '../../checkpoint/mlgcn/VOC2007'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

def main_voc2012() -> object:
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()


    train_dataset = Voc2012Classification(args.data, 'train', inp_name='../../data/VOC2012/voc_glove_word2vec.pkl')
    val_dataset = Voc2012Classification(args.data, 'val', inp_name='../../data/VOC2012/voc_glove_word2vec.pkl')

    num_classes = 20

    # load model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='../../data/VOC2012/voc_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = '../../checkpoint/mlgcn/VOC2012'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

def main_nuswide():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()


    train_dataset = NusWide(args.data, 'train', inp_name='../../data/NUSWIDE/glove_word2vec.pkl')
    val_dataset = NusWide(args.data, 'val', inp_name='../../data/NUSWIDE/glove_word2vec.pkl')

    num_classes = 81

    # load model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='../../data/NUSWIDE/nuswide_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes, 'difficult_examples': True,
             'save_model_path': '../../checkpoint/mlgcn/NUSWIDE', 'workers': args.workers, 'epoch_step': args.epoch_step,
             'lr': args.lr, 'nodes': args.nodes, 'gpus': args.gpus, 'nr': args.nr}
    if args.evaluate:
        state['evaluscrate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

def main_coco():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    train_dataset = COCO2014(args.data, 'train', inp_name='../../data/COCO/coco_glove_word2vec.pkl')
    val_dataset = COCO2014(args.data, 'val', inp_name='../../data/COCO/coco_glove_word2vec.pkl')

    num_classes = 80

    # load model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='../../data/COCO/coco_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes, 'difficult_examples': True,
             'save_model_path': '../../checkpoint/mlgcn/COCO', 'workers': args.workers,
             'epoch_step': args.epoch_step,
             'lr': args.lr, 'nodes': args.nodes, 'gpus': args.gpus, 'nr': args.nr}
    if args.evaluate:
        state['evaluscrate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

def generate_adj_voc2007():
    # for 2007 adj file from train and val dataset
    # for 2012 adj file from train dataset

    result = pickle.load(open('../../data/VOC2007/voc_adj.pkl', 'rb'))

    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
        normalize
    ])
    a = torch.zeros(20)
    b = torch.zeros((20, 20))
    dataset = Voc2007Classification('../../data/VOC2007', 'train', inp_name='../../data/VOC2007/voc_glove_word2vec.pkl')
    dataset.transform = data_transforms
    loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=False,
                                         num_workers=2)
    for (input, target) in loader:
        target[target == -1] = 0
        a = a + torch.sum(target, dim=0)

        for i in range(20):
            for line in target:
                for j in range(20):
                    if line[i] == 1 and line[j] == 1 and i != j:
                        b[i][j] = b[i][j] + 1

    dataset = Voc2007Classification('../../data/VOC2007', 'val', inp_name='../../data/VOC2007/voc_glove_word2vec.pkl')
    dataset.transform = data_transforms
    loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=False,
                                         num_workers=2)
    for (input, target) in loader:
        (img, filename, inp) = input
        target[target == -1] = 0
        # a = a + torch.sum(target, dim=0)

        for i in range(20):
            for line in target:
                for j in range(20):
                    if line[i] == 1 and line[j] == 1 and i != j:
                        b[i][j] = b[i][j] + 1

    print(result)
    print(a)
    print(b)

def generate_adj_voc2012():
    # for 2007 adj file from train and val dataset
    # for 2012 adj file from train dataset
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
        normalize
    ])
    num_class = 20
    a = torch.zeros(num_class)
    b = torch.zeros((num_class, num_class))
    dataset = Voc2012Classification('../../data/VOC2012', 'train', inp_name='../../data/VOC2012/voc_glove_word2vec.pkl')
    dataset.transform = data_transforms
    loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=False,
                                         num_workers=2)

    for (input, target) in loader:
        target[target == -1] = 0
        a = a + torch.sum(target, dim=0)

        for i in range(num_class):
            for line in target:
                for j in range(num_class):
                    if line[i] == 1 and line[j] == 1 and i != j:
                        b[i][j] = b[i][j] + 1

    adj = {}
    adj['nums'] = a.numpy()
    adj['adj'] = b.numpy()
    f = open('../../data/VOC2012/voc_adj.pkl', 'wb')
    pickle.dump(adj, f)
    print(a)
    print(b)

def generate_adj_nuswide():
    # for 2007 adj file from train and val dataset
    # for 2012 adj file from train dataset
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])
    data_transforms = transforms.Compose([
        Warp(448),
        transforms.ToTensor(),
        normalize
    ])
    num_class = 81
    a = torch.zeros(num_class)
    b = torch.zeros((num_class, num_class))
    dataset = NusWide('../../data/NUSWIDE', phase='train', inp_name='../../data/NUSWIDE/glove_word2vec.pkl')
    dataset.transform = data_transforms
    loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=False,
                                         num_workers=2)

    for (input, target) in tqdm(loader):
        target[target == -1] = 0
        a = a + torch.sum(target, dim=0)

        for i in range(num_class):
            for line in target:
                for j in range(num_class):
                    if line[i] == 1 and line[j] == 1 and i != j:
                        b[i][j] = b[i][j] + 1

    adj = {}
    adj['nums'] = a.numpy()
    adj['adj'] = b.numpy()
    f = open('../../data/NUSWIDE/nuswide_adj.pkl', 'wb')
    pickle.dump(adj, f)
    print(a)
    print(b)

def generate_word2vec_nuswide():
    import gensim.downloader as api

    word2vec = np.zeros((81, 300))
    tags = ['airport','animal','beach','bear','birds','boats','book','bridge','buildings','cars','castle','cat','cityscape','clouds','computer','coral','cow','dancing','dog','earthquake','elk','fire','fish','flags','flowers','food','fox','frost','garden','glacier','grass','harbor','horses','house','lake','leaf','map','military','moon','mountain','nighttime','ocean','person','plane','plants','police','protest','railroad','rainbow','reflection','road','rocks','running','sand','sign','sky','snow','soccer','sports','statue','street','sun','sunset','surf','swimmers','tattoo','temple','tiger','tower','town','toy','train','tree','valley','vehicle','water','waterfall','wedding','whales','window','zebra']
    model = api.load("glove-wiki-gigaword-300")
    for i, c in enumerate(tags):
        word2vec[i] = model[c]

    f = open('../../data/NUSWIDE/glove_word2vec.pkl', 'wb')
    pickle.dump(word2vec, f)


if __name__ == '__main__':
    #main_voc2007()
    main_voc2012()
    # main_nuswide()
    # main_coco()
    # generate_word2vec_nuswide()
