import torch
import random
import numpy as np
from nasbench import api
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
    assert args.data in ['cifar10', 'imagenet']
    if args.data == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif args.data == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if args.resize:  # cifar10 or imagenet
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
    else:  # cifar10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    return train_transform, valid_transform


def random_choice(m):
    assert m >= 1
    
    choice = {}
    m_ = np.random.randint(low=1, high=m+1, size=1)[0]
    path_list = random.sample(range(m), m_)
    
    ops = []
    for i in range(m_):
        ops.append(random.sample(range(3), 1)[0])
        # ops.append(random.sample(range(2), 1)[0])

    choice['op'] = ops
    choice['path'] = path_list
    
    return choice


def conv_2_matrix(choice):
    op_ids = choice['op']
    path_ids = choice['path']
    selections = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    
    ops = ['input']
    for i in range(4):  # 初始默认操作
        ops.append(selections[0])
    for i, id in enumerate(path_ids):  # 按choice修改
        ops[id + 1] = selections[op_ids[i]]
    ops.append('conv1x1-bn-relu')
    ops.append('output')
    
    matrix = np.zeros((7, 7), dtype=np.int)
    for id in path_ids:
        matrix[0, id + 1] = 1
        matrix[id + 1, 5] = 1
    matrix[5, -1] = 1
    matrix = matrix.tolist()
    model_spec = api.ModelSpec(matrix=matrix, ops=ops)
    
    return model_spec


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    set_seed(2020)
    for i in range(10):
        choice = random_choice(m=2)
        print(choice)
