import os
import argparse
from utils import *
import torch.nn as nn
from tqdm import tqdm
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model_search import SuperNetwork


def get_args():
    parser = argparse.ArgumentParser("MixPath")
    parser.add_argument('--exp_name', type=str, required=True, help='search model name')
    parser.add_argument('--m', type=int, default=2, required=True, help='num of selected paths as most')
    parser.add_argument('--shadow_bn', action='store_false', default=True, help='shadow bn or not, default: True')
    parser.add_argument('--data_dir', type=str, default='/home/work/dataset/cifar', help='dataset dir')
    parser.add_argument('--classes', type=int, default=10, help='classes')
    parser.add_argument('--layers', type=int, default=12, help='num of MB_layers')
    parser.add_argument('--kernels', type=list, default=[3, 5, 7, 9], help='selective kernels')
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
    parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, imagenet]')
    parser.add_argument('--cutout', action='store_false', default=True, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--colorjitter', action='store_true', default=False, help='use colorjitter')
    arguments = parser.parse_args()

    return arguments


def train(args, epoch, train_data, device, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        choice = random_choice(path_num=len(args.kernels), m=args.m, layers=args.layers)
        outputs = model(inputs, choice)

        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        train_loss += loss.item()

        postfix = {'train loss: {:.6}, train top1: {:.6}, train top5: {:.6}'.format(
            train_loss / (step + 1), top1.avg, top5.avg
        )}
        train_data.set_postfix(log=postfix)


def validate(args, val_data, device, model, choice=None):
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss()
    acc_list = []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            if choice is None:
                choice = random_choice(path_num=len(args.kernels), m=args.m, layers=args.layers)
            outputs = model(inputs, choice)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            val_top5.update(prec5.item(), n)
            acc_list.append(val_top1.avg)

    return val_top1.avg, val_top5.avg, val_loss / (step + 1), acc_list


def main():
    args = get_args()
    print(args)
    # seed
    set_seed(args.seed)

    # prepare dir
    if not os.path.exists('./super_train'):
        os.mkdir('./super_train')
    if not os.path.exists('./super_train/{}'.format(args.exp_name)):
        os.mkdir('./super_train/{}'.format(args.exp_name))

    # device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    criterion = nn.CrossEntropyLoss().to(device)
    model = SuperNetwork(shadow_bn=args.shadow_bn, layers=args.layers, classes=args.classes)
    model = model.to(device)
    print("param size = %fMB" % count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1)

    if args.resume:
        resume_path = './super_train/{}/super_train_states.pt.tar'.format(args.exp_name)
        if os.path.isfile(resume_path):
            print("Loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)

            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            model.load_state_dict(checkpoint['supernet_state'])
            scheduler.laod_state_dict(checkpoint['scheduler_state'])
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

    val_acc_list = []
    for epoch in range(start_epoch, args.epochs):
        # train
        train_data = tqdm(train_queue)
        train_data.set_description(
            '[%s%04d/%04d %s%f]' % ('Epoch:', epoch, args.epochs, 'lr:', scheduler.get_lr()[0]))
        train(args, epoch, train_data, device, model, criterion=criterion, optimizer=optimizer)
        scheduler.step()

        # validate
        val_top1, val_top5, val_obj, val_acc = validate(args, val_data=valid_queue, device=device, model=model)
        val_acc_list.append(val_acc)
        print('val loss: {:.6}, val top1: {:.6}, val top5: {:.6}'.format(val_obj, val_top1, val_top5))
        print(val_acc)

        # save the states of this epoch
        state = {
            'epoch': epoch,
            'args': args,
            'optimizer_state': optimizer.state_dict(),
            'supernet_state': model.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        path = './super_train/{}/super_train_states.pt.tar'.format(args.exp_name)
        torch.save(state, path)
    print(val_acc_list)


if __name__ == '__main__':
    main()
