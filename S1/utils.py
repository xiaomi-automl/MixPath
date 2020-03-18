import time
import torch
import random
import numpy as np
import collections
from torch.autograd import Variable
import torchvision.transforms as transforms


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def data_transforms_cifar(args):
    assert args.dataset in ['cifar10', 'imagenet']
    if args.dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif args.dataset == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if args.dataset == 'cifar10':
        random_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()]
        if args.colorjitter:
            random_transform += [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)]
        normalize_transform = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)]
        train_transform = transforms.Compose(
            random_transform + normalize_transform
        )
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    elif args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    return train_transform, valid_transform


def random_choice(path_num, m, layers):
    # choice = {}
    choice = collections.OrderedDict()
    for i in range(layers):
        # expansion rate
        rate = np.random.randint(low=0, high=2, size=1)[0]
        # conv
        m_ = np.random.randint(low=1, high=(m+1), size=1)[0]
        rand_conv = random.sample(range(path_num), m_)
        choice[i] = {'conv': rand_conv, 'rate': rate}
    return choice


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def eta_time(elapse, epoch):
    eta = epoch * elapse
    hour = eta // 3600
    minute = (eta - hour * 3600) // 60
    second = eta - hour * 3600 - minute * 60
    return hour, minute, second


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed: hour: %d, minute: %d, second: %f' % (hour, minute, second))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        if str(x.device) == 'cpu':
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    for i in range(8):
        np.random.seed(12)
        random.seed(12)
        choice = random_choice(path_num=3, m=2, layers=12)
        print(choice)
