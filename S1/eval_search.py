import os
import numpy as np
import argparse
from utils import *
import autograd.numpy as anp
import pymoo
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
#from wnsga2 import WNSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.visualization.scatter import Scatter
from model_search import SuperNetwork
import torch
import time
import math
from itertools import combinations
import collections
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tqdm import tqdm


@torch.no_grad()
def naive_latency(model, choice, size:tuple)->float:
    tt = time.time()
    device = next(model.parameters()).device
    input = torch.rand(32, 3, *size).to(device)
    for i in range(10):
        model(input, choice)
    return (time.time() - tt)/10


def combine(n, m):
    assert n >= m
    f = math.factorial
    return int(f(n)/f(m)/f(n-m))


def get_choice_list(path_num, m, rate_num=2):
    assert path_num >= m
    conv_choices = []
    for i in range(1, m+1):
        conv_choices.extend(
            combinations(range(path_num), i))
    rate_choices = list(range(rate_num))
    choices = []
    for i in range(rate_num):
        for conv_choice in conv_choices:
            choices.append(
                dict( conv=conv_choice, rate=i )
            )
    return choices


class MyProblem(Problem):
    def __init__(self, model, valid_queue, device, choices, layers):
        self.model = model
        self.valid_queue = valid_queue
        self.device = device
        self.choices = choices
        self.generate = 0
        super().__init__(n_var=layers,
                         n_obj=3,
                         n_constr=0,
                         xl=anp.array([0 for i in range(layers)]),
                         xu=anp.array([len(choices)-1 for i in range(layers)]))

    def intarray2choice(self, x):
        choice = collections.OrderedDict()
        for i in range(len(x)):
            c = self.choices[x[i]]
            choice[i] = c
        return choice

    def _evaluate(self, x, out, *args, **kwargs):
        """
        max acc, parameters
        min latency
        """
        num_pop = x.shape[0]
        f1 = np.zeros(num_pop)
        f2 = np.zeros(num_pop)
        f3 = np.zeros(num_pop)
        for i in range(num_pop):
            choice = self.intarray2choice(x[i])
            acc = self.get_accuracy(choice)
            f1[i] = 1.0 - acc
            para_amount = self.get_para_amount(choice)
            f2[i] = - para_amount
            latency = self.get_latency(choice)
            f3[i] = latency
        out["F"] = anp.column_stack([f1, f2, f3])
        self.generate += 1

    @torch.no_grad()
    def get_accuracy(self, choice):
        if choice is None:
            assert False
        self.model.eval()
        all_targets = []
        all_outputs = []
        for step, (inputs, targets) in tqdm(enumerate(self.valid_queue), total=len(self.valid_queue)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            all_targets.append(targets)
            outputs = self.model(inputs, choice)
            all_outputs.append(outputs)
        all_targets = torch.cat(all_targets)
        all_outputs = torch.cat(all_outputs)
        prec1 = accuracy(all_outputs, all_targets, topk=(1,))
        return prec1[0].cpu().item()

    def get_para_amount(self, choice):
        return count_parameters_in_MB(self.model.get_submodule(choice))

    def get_latency(self, choice):
        return naive_latency(self.model, choice, size=(32, 32))
        # TODO calculate latency based on
        # latency lookup table for SNPE, OPENVINO, etc


def get_args():
    parser = argparse.ArgumentParser("Search The MixPath")
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
    #*************GA*****************#
    parser.add_argument('--model_path' , type=str, required=True)
    parser.add_argument('--pop_size', type=int, default=40)
    parser.add_argument('--n_offsprings', type=int, default=10)
    parser.add_argument('--n_generations', type=int, default=40)

    arguments = parser.parse_args()


    return arguments


def main():
    args = get_args()
    print(args)

    # prepare dir
    if not os.path.exists('./super_train'):
        os.mkdir('./super_train')
    if not os.path.exists('./super_train/{}'.format(args.exp_name)):
        save_path = './super_train/{}'.format(args.exp_name)
        os.mkdir(save_path)

    # device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")


    model = SuperNetwork(shadow_bn=args.shadow_bn, layers=args.layers, classes=args.classes)
    model = model.to(device)
    print("param size of supernet = %fMB" % count_parameters_in_MB(model))
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['supernet_state'])
    train_transform, valid_transform = data_transforms_cifar(args)
    valset = dset.CIFAR10(root=args.data_dir, train=False, download=False, transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True, num_workers=8)

    choices = get_choice_list(path_num=len(args.kernels), m=args.m)
    problem = MyProblem(model, valid_queue, device, choices, args.layers)

    algorithm = NSGA2(
        pop_size=args.pop_size,
        n_offsprings=args.n_offsprings,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_one_point"),
        mutation=get_mutation("int_pm"),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", args.n_generations)

    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               pf=problem.pareto_front(use_cache=False),
               save_history=True,
               verbose=True)
    print(res)
    save_path = './super_train/{}'.format(args.exp_name)
    torch.save(res, f"{save_path}/res.pkl")
    # TODO plot 3D pareto_front points
    ## get the pareto-set and pareto-front for plotting
    #ps = problem.pareto_set(use_cache=False, flatten=False)
    #pf = problem.pareto_front(use_cache=False, flatten=False)

    ## Design Space
    #plot = Scatter(title = "Design Space", axis_labels="x")
    #plot.add(res.X, s=30, facecolors='none', edgecolors='r')
    #plot.add(ps, plot_type="line", color="black", alpha=0.7)
    #plot.do()
    #plot.apply(lambda ax: ax.set_xlim(-0.5, 1.5))
    #plot.apply(lambda ax: ax.set_ylim(-2, 2))
    ##plot.show()
    #plot.savefig(f"{save_path}/design_space.png")

    ## Objective Space
    #plot = Scatter(title = "Objective Space")
    #plot.add(res.F)
    #plot.add(pf, plot_type="line", color="black", alpha=0.7)
    ##plot.show()
    #plot.savefig(f"{save_path}/objective_space.png")


if __name__ == "__main__":
    main()
