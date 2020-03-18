import os
import sys
import ast
import argparse
import nas_utils
from tqdm import tqdm
import torch.nn as nn
from nas_utils import *
from scipy.stats import kendalltau
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from nas_model_search import SuperNetwork


def get_args():
    parser = argparse.ArgumentParser("MixPath")
    parser.add_argument('--exp_name', type=str, required=True, help='search model name')
    parser.add_argument('--m', type=int, default=2, required=True, help='num of selected paths as most')
    parser.add_argument('--shadow_bn', action='store_false', default=True, help='shadow bn or not, default: True')
    parser.add_argument('--data_dir', type=str, default='/home/work/dataset/cifar', help='dataset dir')
    parser.add_argument('--classes', type=int, default=10, help='classes')
    parser.add_argument('--layers', type=int, default=12, help='num of MB_layers')
    parser.add_argument('--kernels', type=list, default=[3, 5, 7], help='selective kernels')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='num of epochs')
    parser.add_argument('--seed', type=int, default=2020, help='seed')
    parser.add_argument('--search_num', type=int, default=1000, help='num of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-8, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--train_interval', type=int, default=1, help='train to print frequency')
    parser.add_argument('--val_interval', type=int, default=5, help='evaluate and save frequency')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='drop out rate')
    parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop_path_prob')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', type=bool, default=False, help='resume')
    # ******************************* dataset *******************************#
    parser.add_argument('--data', type=str, default='cifar10', help='[cifar10, imagenet]')
    parser.add_argument('--cutout', action='store_false', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--resize', action='store_true', default=False, help='use resize')

    arguments = parser.parse_args()

    return arguments


def validate_cali(args, val_data, device, model, choice):
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, choice)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            val_top5.update(prec5.item(), n)
    print(val_top1.avg, ',')
    return val_top1.avg, val_top5.avg, val_loss / (step + 1)

check_dict = []
def validate_search(args, val_data, device, model):
    model.eval()
    choice_dict = {}
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss()
    choice = random_choice(m=args.m)

    while choice in check_dict:
        print('Duplicate Index !')
        choice = random_choice(m=args.m)
    check_dict.append(choice)
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            val_top5.update(prec5.item(), n)
    choice_dict['op'] = choice['op']
    choice_dict['path'] = choice['path']
    choice_dict['val_loss'] = val_loss / (step + 1)
    choice_dict['val_top1'] = val_top1.avg

    return choice_dict


def train(args, epoch, train_data, device, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        choice = random_choice(m=args.m)
        outputs = model(inputs, choice)

        loss = criterion(outputs, targets)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        train_loss += loss.item()

        postfix = {'loss': '%.6f' % (train_loss / (step + 1)), 'top1': '%.3f' % top1.avg}

        train_data.set_postfix(postfix)


def validate(args, val_data, device, model):
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        top1_m = []
        top5_m = []
        loss_m = []
        for _ in range(20):
            choice = random_choice(m=args.m)
            for step, (inputs, targets) in enumerate(val_data):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, choice)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                n = inputs.size(0)
                val_top1.update(prec1.item(), n)
                val_top5.update(prec5.item(), n)
            top1_m.append(val_top1.avg), top5_m.append(val_top5.avg), loss_m.append(val_loss / (step + 1))

    return np.mean(top1_m), np.mean(top5_m), np.mean(loss_m)


def separate_bn_params(model):
    bn_index = []
    bn_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_index += list(map(id, m.parameters()))
            bn_params += m.parameters()
    base_params = list(filter(lambda p: id(p) not in bn_index, model.parameters()))
    return base_params, bn_params


def main():
    args = get_args()
    print(args)

    if not os.path.exists('./snapshots'):
        os.mkdir('./snapshots')

    # device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    set_seed(args.seed)

    criterion = nn.CrossEntropyLoss()
    model = SuperNetwork(init_channels=128, shadow_bn=args.shadow_bn)
    model = model.to(device)
    print("param size = %fMB" % count_parameters_in_MB(model))

    base_params, bn_params = separate_bn_params(model)

    optimizer = torch.optim.SGD([
        {'params': base_params, 'weight_decay': args.weight_decay},
        {'params': bn_params, 'weight_decay': 0.0}],
        lr=args.learning_rate,
        momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1)

    if args.resume:
        resume_path = './snapshots/{}_states.pt.tar'.format(args.exp_name)
        if os.path.isfile(resume_path):
            print("Loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)

            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            model.load_state_dict(checkpoint['supernet_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        else:
            raise ValueError("No checkpoint found at '{}'".format(resume_path))
    else:
        start_epoch = 0

    train_transform, valid_transform = data_transforms_cifar(args)
    trainset = dset.CIFAR10(root=args.data_dir, train=True, download=False, transform=train_transform)
    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, pin_memory=True, num_workers=8)
    valset = dset.CIFAR10(root=args.data_dir, train=False, download=False, transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=8)

    top1 = []
    top5 = []
    loss = []
    for epoch in range(start_epoch, args.epochs):
        train_data = tqdm(train_queue)
        train_data.set_description(
            '[%s%04d/%04d %s%f]' % ('Epoch:', epoch+1, args.epochs, 'lr:', scheduler.get_lr()[0]))
        train(args, epoch, train_data, device, model, criterion=criterion, optimizer=optimizer)
        scheduler.step()

        if epoch % 2 == 1:
            # validation the model
            val_top1, val_top5, val_loss = validate(args, val_data=valid_queue, device=device, model=model)
            print('val loss: {:.6}, val top1: {:.6}'.format(val_loss, val_top1))

            # save the states of this epoch
            state = {
                'epoch': epoch,
                'args': args,
                'optimizer_state': optimizer.state_dict(),
                'supernet_state': model.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            path = './snapshots/{}_states.pt.tar'.format(args.exp_name)
            torch.save(state, path)
            top1.append(val_top1), top5.append(val_top5), loss.append(val_loss)
    print('top1:', top1)
    print('top5:', top5)
    print('loss:', loss)


    candidate_dict = {}
    for epoch in range(args.search_num):
        # validation
        choice_dict = validate_search(args, val_data=valid_queue, device=device, model=model)
        candidate_dict[str(choice_dict)] = choice_dict['val_top1']
        print('epoch: {:d},val loss: {:.6}, val top1: {:.6}'.format(
            epoch, choice_dict['val_loss'], choice_dict['val_top1']))
    print(candidate_dict)

    # sort candidate_dict
    print('\n', '****************************** supernet *********************************')
    cand_dict = {k: v for k, v in sorted(candidate_dict.items(), key=lambda item: item[1])}
    for key in cand_dict.keys():
        key = ast.literal_eval(key)
        print(key['val_top1'], ',')

    # look-up nasbench
    print('\n', '****************************** nas_bench *********************************')
    NASBENCH_TFRECORD = './nasbench_only108.tfrecord'
    nasbench = api.NASBench(NASBENCH_TFRECORD)
    nasbench_acc = []
    for key in cand_dict.keys():
        key = ast.literal_eval(key)
        choice = {}
        choice['op'] = key['op']
        choice['path'] = key['path']
        model_spec = nas_utils.conv_2_matrix(choice)
        data = nasbench.query(model_spec)
        nasbench_acc.append(data['validation_accuracy'])
        print(data['validation_accuracy'], ',')

    print('\n', '****************************** supernet **********************************')
    supernet_acc = []
    for key in cand_dict.keys():
        key = ast.literal_eval(key)
        supernet_acc.append(key['val_top1'])
        print(key['val_top1'], ',')

    # cali_bn
    print('\n', '****************************** cali_bn **********************************')
    cali_bn_acc = []
    checkpoint = torch.load('./snapshots/{}_states.pt.tar'.format(args.exp_name))
    for key in cand_dict.keys():
        with torch.no_grad():
            choice = {}
            key = ast.literal_eval(key)
            choice['op'] = key['op']
            choice['path'] = key['path']
            model.train()
            for inputs, targets in valid_queue:
                inputs, targets = inputs.to(device), targets.to(device)
                model(inputs, choice)
            top1_acc, _, _ = validate_cali(args, valid_queue, device, model, choice)
            cali_bn_acc.append(top1_acc)
            model.load_state_dict(checkpoint['supernet_state'])

    # ranking
    print('\n', '****************************** ranking **********************************')
    print('before_cali:', kendalltau(supernet_acc[30:], nasbench_acc[30:]))
    print('after_cali:', kendalltau(cali_bn_acc[30:], nasbench_acc[30:]))


if __name__ == '__main__':
    main()
