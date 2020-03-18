import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nas_utils import random_choice


class ConvBnRelu(nn.Module):
	def __init__(self, inplanes, outplanes, k):
		super(ConvBnRelu, self).__init__()
		
		self.op = nn.Sequential(
			nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(outplanes),
			nn.ReLU(),
			
			nn.Conv2d(outplanes, outplanes, kernel_size=k, stride=1, padding=k // 2, bias=False),
			nn.BatchNorm2d(outplanes),
			nn.ReLU()
		)
	
	def forward(self, x):
		return self.op(x)


class MaxPool(nn.Module):
	def __init__(self, inplanes, outplanes):
		super(MaxPool, self).__init__()
		
		self.op = nn.Sequential(
			nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(outplanes),
			nn.ReLU(),
			
			nn.MaxPool2d(3, 1, padding=1)
		)
	
	def forward(self, x):
		return self.op(x)


class Cell(nn.Module):
	def __init__(self, inplanes, outplanes, shadow_bn):
		super(Cell, self).__init__()
		self.inplanes = inplanes
		self.outplanes = outplanes
		self.shadow_bn = shadow_bn

		self.nodes = nn.ModuleList([])
		for i in range(4):
			self.nodes.append(ConvBnRelu(self.inplanes, self.outplanes, 1))
			self.nodes.append(ConvBnRelu(self.inplanes, self.outplanes, 3))
			self.nodes.append(MaxPool(self.inplanes, self.outplanes))
		self.nodes.append(nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1))

		self.bn_list = nn.ModuleList([])
		if self.shadow_bn:
			for j in range(4):
				self.bn_list.append(nn.BatchNorm2d(outplanes))
		else:
			self.bn = nn.BatchNorm2d(outplanes)

	def forward(self, x, choice):
		path_ids = choice['path']       # eg.[0, 2, 3]
		op_ids = choice['op']   # eg.[1, 1, 2]
		x_list = []
		for i, id in enumerate(path_ids):
			x_list.append(self.nodes[id * 3 + op_ids[i]](x))
		
		x = sum(x_list)
		out = self.nodes[-1](x)
		return F.relu(out)


class SuperNetwork(nn.Module):
	def __init__(self, init_channels, classes=10, shadow_bn=True):
		super(SuperNetwork, self).__init__()
		self.init_channels = init_channels
		
		self.stem = nn.Sequential(
			nn.Conv2d(3, self.init_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(self.init_channels),
			nn.ReLU(inplace=True)
		)
		
		self.cell_list = nn.ModuleList([])
		for i in range(9):
			if i in [3, 6]:
				self.cell_list.append(Cell(self.init_channels, self.init_channels * 2, shadow_bn=shadow_bn))
				self.init_channels *= 2
			else:
				self.cell_list.append(Cell(self.init_channels, self.init_channels, shadow_bn=shadow_bn))

		self.global_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(self.init_channels, classes)
		self._initialize_weights()
	
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1.0)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(0)  # fan-out
				init_range = 1.0 / math.sqrt(n)
				m.weight.data.uniform_(-init_range, init_range)
				m.bias.data.zero_()
	
	def forward(self, x, choice):
		x = self.stem(x)
		for i in range(9):
			x = self.cell_list[i](x, choice)
			if i in [2, 5]:
				x = nn.MaxPool2d(2, 2, padding=0)(x)
		x = self.global_pooling(x)
		x = x.view(-1, self.init_channels)
		out = self.classifier(x)
		
		return out


if __name__ == '__main__':
	# ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
	# choice = {'path': [0, 1, 2],  # a list of shape (4, )
	#           'op': [0, 0, 0]}  # possible shapes: (), (1, ), (2, ), (3, )
	choice = random_choice(3)
	print(choice)
	model = SuperNetwork(init_channels=128)
	input = torch.randn((1, 3, 32, 32))
	print(model(input, choice))
