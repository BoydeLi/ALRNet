import torch
import torch.nn as nn
from search.operations import *
import torchvision.models as models
from torch.autograd import Variable
from search.utils import drop_path
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from model.Res2Net_v1b import res2net50_v1b_26w_4s


class CoarseAggreCell_1(nn.Module): ##ACLM34  steps=4 multiplier=4
  def __init__(self, genotype, steps, multiplier, C, parse_method):
    super(CoarseAggreCell_1, self).__init__()
    op_names, indices = zip(*genotype.CoarseAggre_1)
    concat = genotype.CoarseAggre_1_concat

    self.f3_preprocess = nn.Sequential(
      nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True))

    self.f4_preprocess = nn.Sequential(
    )

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, f3, f4, drop_prob):
    f3 = self.f3_preprocess(f3)
    f4 = self.f4_preprocess(f4)

    states = [f3, f4]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]

    return torch.cat([states[i] for i in self._concat], dim=1)  #f34

class CoarseAggreCell_2(nn.Module): ##ACLM45  steps=4 multiplier=4
  def __init__(self, genotype, steps, multiplier, C, parse_method):
    super(CoarseAggreCell_2, self).__init__()
    op_names, indices = zip(*genotype.CoarseAggre_2)
    concat = genotype.CoarseAggre_2_concat

    self.f34_preprocess = nn.Sequential(
      nn.Conv2d(256, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True),
      nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True))

    self.f5_preprocess = nn.Sequential(
    )

    self.output_preprocess = nn.Sequential(
      nn.Conv2d(256, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True),
    )

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, f34, f5, drop_prob):
    f34 = self.f34_preprocess(f34)
    f5 = self.f5_preprocess(f5)

    states = [f34, f5]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    output = torch.cat([states[i] for i in self._concat], dim=1)
    output = self.output_preprocess(output)

    return output


class AttentionCell_1(nn.Module): #ARFB1 steps=2 multiplier=2
  def __init__(self, genotype, steps, multiplier, Cin, C, parse_method):
    super(AttentionCell_1, self).__init__()
    op_names_b1, indices_b1 = zip(*genotype.Attention1_b1)
    concat = genotype.Attention1_b1_concat
    op_names_b2, indices_b2 = zip(*genotype.Attention1_b2)
    op_names_b3, indices_b3 = zip(*genotype.Attention1_b3)
    op_names_b4, indices_b4 = zip(*genotype.Attention1_b4)

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat)

    self.branch1_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch5_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.branch1_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.f_att_preprocess = nn.Sequential(
      nn.Conv2d(4 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

  def _compile(self, C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat):
    assert len(op_names_b1) == len(indices_b1)
    assert len(op_names_b2) == len(indices_b2)
    assert len(op_names_b3) == len(indices_b3)
    assert len(op_names_b4) == len(indices_b4)
    self._concat = concat
    self.multiplier = len(concat)

    self.b1_ops = nn.ModuleList()
    self.b2_ops = nn.ModuleList()
    self.b3_ops = nn.ModuleList()
    self.b4_ops = nn.ModuleList()

    for name, index in zip(op_names_b1, indices_b1):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b1_ops += [op]
    self.b1_indices = indices_b1

    for name, index in zip(op_names_b2, indices_b2):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b2_ops += [op]
    self.b2_indices = indices_b2

    for name, index in zip(op_names_b3, indices_b3):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b3_ops += [op]
    self.b3_indices = indices_b3

    for name, index in zip(op_names_b4, indices_b4):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b4_ops += [op]
    self.b4_indices = indices_b4

  def forward(self, fi, drop_prob):
    branch1_fi = self.branch1_preprocess(fi)
    branch2_fi = self.branch2_preprocess(fi)
    branch3_fi = self.branch3_preprocess(fi)
    branch4_fi = self.branch4_preprocess(fi)
    branch5_fi = self.branch5_preprocess(fi)

    branch1_states = [branch1_fi]
    branch2_states = [branch2_fi]
    branch3_states = [branch3_fi]
    branch4_states = [branch4_fi]
    for i in range(self._steps):
      h1 = branch1_states[self.b1_indices[1 * i]]
      op1 = self.b1_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch1_states += [s]

      h1 = branch2_states[self.b2_indices[1 * i]]
      op1 = self.b2_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch2_states += [s]

      h1 = branch3_states[self.b3_indices[1 * i]]
      op1 = self.b3_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch3_states += [s]

      h1 = branch4_states[self.b4_indices[1 * i]]
      op1 = self.b4_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch4_states += [s]

    branch1_output = torch.cat([branch1_states[i] for i in self._concat], dim=1)
    branch2_output = torch.cat([branch2_states[i] for i in self._concat], dim=1)
    branch3_output = torch.cat([branch3_states[i] for i in self._concat], dim=1)
    branch4_output = torch.cat([branch4_states[i] for i in self._concat], dim=1)
    branch1_output = self.branch1_att_preprocess(branch1_output)
    branch2_output = self.branch2_att_preprocess(branch2_output)
    branch3_output = self.branch3_att_preprocess(branch3_output)
    branch4_output = self.branch4_att_preprocess(branch4_output)

    output = torch.cat((branch1_output, branch2_output, branch3_output, branch4_output), dim=1)
    output = self.f_att_preprocess(output)
    output = branch5_fi + output

    return output


class AttentionCell_2(nn.Module):  # ARFB2 steps=2 multiplier=2
  def __init__(self, genotype, steps, multiplier, Cin, C, parse_method):
    super(AttentionCell_2, self).__init__()
    op_names_b1, indices_b1 = zip(*genotype.Attention2_b1)
    concat = genotype.Attention2_b1_concat
    op_names_b2, indices_b2 = zip(*genotype.Attention2_b2)
    op_names_b3, indices_b3 = zip(*genotype.Attention2_b3)
    op_names_b4, indices_b4 = zip(*genotype.Attention2_b4)

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat)

    self.branch1_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch5_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.branch1_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.f_att_preprocess = nn.Sequential(
      nn.Conv2d(4 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

  def _compile(self, C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat):
    assert len(op_names_b1) == len(indices_b1)
    assert len(op_names_b2) == len(indices_b2)
    assert len(op_names_b3) == len(indices_b3)
    assert len(op_names_b4) == len(indices_b4)
    self._concat = concat
    self.multiplier = len(concat)

    self.b1_ops = nn.ModuleList()
    self.b2_ops = nn.ModuleList()
    self.b3_ops = nn.ModuleList()
    self.b4_ops = nn.ModuleList()

    for name, index in zip(op_names_b1, indices_b1):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b1_ops += [op]
    self.b1_indices = indices_b1

    for name, index in zip(op_names_b2, indices_b2):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b2_ops += [op]
    self.b2_indices = indices_b2

    for name, index in zip(op_names_b3, indices_b3):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b3_ops += [op]
    self.b3_indices = indices_b3

    for name, index in zip(op_names_b4, indices_b4):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b4_ops += [op]
    self.b4_indices = indices_b4

  def forward(self, fi, drop_prob):
    branch1_fi = self.branch1_preprocess(fi)
    branch2_fi = self.branch2_preprocess(fi)
    branch3_fi = self.branch3_preprocess(fi)
    branch4_fi = self.branch4_preprocess(fi)
    branch5_fi = self.branch5_preprocess(fi)

    branch1_states = [branch1_fi]
    branch2_states = [branch2_fi]
    branch3_states = [branch3_fi]
    branch4_states = [branch4_fi]
    for i in range(self._steps):
      h1 = branch1_states[self.b1_indices[1 * i]]
      op1 = self.b1_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch1_states += [s]

      h1 = branch2_states[self.b2_indices[1 * i]]
      op1 = self.b2_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch2_states += [s]

      h1 = branch3_states[self.b3_indices[1 * i]]
      op1 = self.b3_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch3_states += [s]

      h1 = branch4_states[self.b4_indices[1 * i]]
      op1 = self.b4_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch4_states += [s]

    branch1_output = torch.cat([branch1_states[i] for i in self._concat], dim=1)
    branch2_output = torch.cat([branch2_states[i] for i in self._concat], dim=1)
    branch3_output = torch.cat([branch3_states[i] for i in self._concat], dim=1)
    branch4_output = torch.cat([branch4_states[i] for i in self._concat], dim=1)
    branch1_output = self.branch1_att_preprocess(branch1_output)
    branch2_output = self.branch2_att_preprocess(branch2_output)
    branch3_output = self.branch3_att_preprocess(branch3_output)
    branch4_output = self.branch4_att_preprocess(branch4_output)

    output = torch.cat((branch1_output, branch2_output, branch3_output, branch4_output), dim=1)
    output = self.f_att_preprocess(output)
    output = branch5_fi + output

    return output


class AttentionCell_3(nn.Module):  # ARFB3 steps=2 multiplier=2
  def __init__(self, genotype, steps, multiplier, Cin, C, parse_method):
    super(AttentionCell_3, self).__init__()
    op_names_b1, indices_b1 = zip(*genotype.Attention3_b1)
    concat = genotype.Attention3_b1_concat
    op_names_b2, indices_b2 = zip(*genotype.Attention3_b2)

    op_names_b3, indices_b3 = zip(*genotype.Attention3_b3)

    op_names_b4, indices_b4 = zip(*genotype.Attention3_b4)

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat)

    self.branch1_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch5_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.branch1_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.f_att_preprocess = nn.Sequential(
      nn.Conv2d(4 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

  def _compile(self, C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4,
               indices_b4, concat):
    assert len(op_names_b1) == len(indices_b1)
    assert len(op_names_b2) == len(indices_b2)
    assert len(op_names_b3) == len(indices_b3)
    assert len(op_names_b4) == len(indices_b4)
    self._concat = concat
    self.multiplier = len(concat)

    self.b1_ops = nn.ModuleList()
    self.b2_ops = nn.ModuleList()
    self.b3_ops = nn.ModuleList()
    self.b4_ops = nn.ModuleList()

    for name, index in zip(op_names_b1, indices_b1):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b1_ops += [op]
    self.b1_indices = indices_b1

    for name, index in zip(op_names_b2, indices_b2):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b2_ops += [op]
    self.b2_indices = indices_b2

    for name, index in zip(op_names_b3, indices_b3):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b3_ops += [op]
    self.b3_indices = indices_b3

    for name, index in zip(op_names_b4, indices_b4):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b4_ops += [op]
    self.b4_indices = indices_b4

  def forward(self, fi, drop_prob):
    branch1_fi = self.branch1_preprocess(fi)
    branch2_fi = self.branch2_preprocess(fi)
    branch3_fi = self.branch3_preprocess(fi)
    branch4_fi = self.branch4_preprocess(fi)
    branch5_fi = self.branch5_preprocess(fi)

    branch1_states = [branch1_fi]
    branch2_states = [branch2_fi]
    branch3_states = [branch3_fi]
    branch4_states = [branch4_fi]
    for i in range(self._steps):
      h1 = branch1_states[self.b1_indices[1 * i]]
      op1 = self.b1_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch1_states += [s]

      h1 = branch2_states[self.b2_indices[1 * i]]
      op1 = self.b2_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch2_states += [s]

      h1 = branch3_states[self.b3_indices[1 * i]]
      op1 = self.b3_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch3_states += [s]

      h1 = branch4_states[self.b4_indices[1 * i]]
      op1 = self.b4_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch4_states += [s]

    branch1_output = torch.cat([branch1_states[i] for i in self._concat], dim=1)
    branch2_output = torch.cat([branch2_states[i] for i in self._concat], dim=1)
    branch3_output = torch.cat([branch3_states[i] for i in self._concat], dim=1)
    branch4_output = torch.cat([branch4_states[i] for i in self._concat], dim=1)
    branch1_output = self.branch1_att_preprocess(branch1_output)
    branch2_output = self.branch2_att_preprocess(branch2_output)
    branch3_output = self.branch3_att_preprocess(branch3_output)
    branch4_output = self.branch4_att_preprocess(branch4_output)

    output = torch.cat((branch1_output, branch2_output, branch3_output, branch4_output), dim=1)
    output = self.f_att_preprocess(output)
    output = branch5_fi + output

    return output


class AttentionCell_4(nn.Module):  # ARFB4 steps=2 multiplier=2
  def __init__(self, genotype, steps, multiplier, Cin, C, parse_method):
    super(AttentionCell_4, self).__init__()
    op_names_b1, indices_b1 = zip(*genotype.Attention4_b1)
    concat = genotype.Attention4_b1_concat
    op_names_b2, indices_b2 = zip(*genotype.Attention4_b2)
    op_names_b3, indices_b3 = zip(*genotype.Attention4_b3)
    op_names_b4, indices_b4 = zip(*genotype.Attention4_b4)

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat)

    self.branch1_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch5_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.branch1_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.f_att_preprocess = nn.Sequential(
      nn.Conv2d(4 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

  def _compile(self, C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat):
    assert len(op_names_b1) == len(indices_b1)
    assert len(op_names_b2) == len(indices_b2)
    assert len(op_names_b3) == len(indices_b3)
    assert len(op_names_b4) == len(indices_b4)
    self._concat = concat
    self.multiplier = len(concat)

    self.b1_ops = nn.ModuleList()
    self.b2_ops = nn.ModuleList()
    self.b3_ops = nn.ModuleList()
    self.b4_ops = nn.ModuleList()

    for name, index in zip(op_names_b1, indices_b1):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b1_ops += [op]
    self.b1_indices = indices_b1

    for name, index in zip(op_names_b2, indices_b2):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b2_ops += [op]
    self.b2_indices = indices_b2

    for name, index in zip(op_names_b3, indices_b3):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b3_ops += [op]
    self.b3_indices = indices_b3

    for name, index in zip(op_names_b4, indices_b4):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b4_ops += [op]
    self.b4_indices = indices_b4

  def forward(self, fi, drop_prob):
    branch1_fi = self.branch1_preprocess(fi)
    branch2_fi = self.branch2_preprocess(fi)
    branch3_fi = self.branch3_preprocess(fi)
    branch4_fi = self.branch4_preprocess(fi)
    branch5_fi = self.branch5_preprocess(fi)

    branch1_states = [branch1_fi]
    branch2_states = [branch2_fi]
    branch3_states = [branch3_fi]
    branch4_states = [branch4_fi]
    for i in range(self._steps):
      h1 = branch1_states[self.b1_indices[1 * i]]
      op1 = self.b1_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch1_states += [s]

      h1 = branch2_states[self.b2_indices[1 * i]]
      op1 = self.b2_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch2_states += [s]

      h1 = branch3_states[self.b3_indices[1 * i]]
      op1 = self.b3_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch3_states += [s]

      h1 = branch4_states[self.b4_indices[1 * i]]
      op1 = self.b4_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch4_states += [s]

    branch1_output = torch.cat([branch1_states[i] for i in self._concat], dim=1)
    branch2_output = torch.cat([branch2_states[i] for i in self._concat], dim=1)
    branch3_output = torch.cat([branch3_states[i] for i in self._concat], dim=1)
    branch4_output = torch.cat([branch4_states[i] for i in self._concat], dim=1)
    branch1_output = self.branch1_att_preprocess(branch1_output)
    branch2_output = self.branch2_att_preprocess(branch2_output)
    branch3_output = self.branch3_att_preprocess(branch3_output)
    branch4_output = self.branch4_att_preprocess(branch4_output)

    output = torch.cat((branch1_output, branch2_output, branch3_output, branch4_output), dim=1)
    output = self.f_att_preprocess(output)
    output = branch5_fi + output

    return output


class AttentionCell_5(nn.Module):  # ARFB5 steps=2 multiplier=2
  def __init__(self, genotype, steps, multiplier, Cin, C, parse_method):
    super(AttentionCell_5, self).__init__()
    op_names_b1, indices_b1 = zip(*genotype.Attention5_b1)
    concat = genotype.Attention5_b1_concat
    op_names_b2, indices_b2 = zip(*genotype.Attention5_b2)
    op_names_b3, indices_b3 = zip(*genotype.Attention5_b3)
    op_names_b4, indices_b4 = zip(*genotype.Attention5_b4)

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat)

    self.branch1_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch5_preprocess = nn.Sequential(
      nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.branch1_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch2_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch3_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.branch4_att_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.f_att_preprocess = nn.Sequential(
      nn.Conv2d(4 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

  def _compile(self, C, op_names_b1, indices_b1, op_names_b2, indices_b2, op_names_b3, indices_b3, op_names_b4, indices_b4, concat):
    assert len(op_names_b1) == len(indices_b1)
    assert len(op_names_b2) == len(indices_b2)
    assert len(op_names_b3) == len(indices_b3)
    assert len(op_names_b4) == len(indices_b4)
    self._concat = concat
    self.multiplier = len(concat)

    self.b1_ops = nn.ModuleList()
    self.b2_ops = nn.ModuleList()
    self.b3_ops = nn.ModuleList()
    self.b4_ops = nn.ModuleList()

    for name, index in zip(op_names_b1, indices_b1):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b1_ops += [op]
    self.b1_indices = indices_b1

    for name, index in zip(op_names_b2, indices_b2):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b2_ops += [op]
    self.b2_indices = indices_b2

    for name, index in zip(op_names_b3, indices_b3):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b3_ops += [op]
    self.b3_indices = indices_b3

    for name, index in zip(op_names_b4, indices_b4):
      stride = 1
      op = OPS[name](C, stride, True)
      self.b4_ops += [op]
    self.b4_indices = indices_b4

  def forward(self, fi, drop_prob):
    branch1_fi = self.branch1_preprocess(fi)
    branch2_fi = self.branch2_preprocess(fi)
    branch3_fi = self.branch3_preprocess(fi)
    branch4_fi = self.branch4_preprocess(fi)
    branch5_fi = self.branch5_preprocess(fi)

    branch1_states = [branch1_fi]
    branch2_states = [branch2_fi]
    branch3_states = [branch3_fi]
    branch4_states = [branch4_fi]
    for i in range(self._steps):
      h1 = branch1_states[self.b1_indices[1 * i]]
      op1 = self.b1_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch1_states += [s]

      h1 = branch2_states[self.b2_indices[1 * i]]
      op1 = self.b2_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch2_states += [s]

      h1 = branch3_states[self.b3_indices[1 * i]]
      op1 = self.b3_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch3_states += [s]

      h1 = branch4_states[self.b4_indices[1 * i]]
      op1 = self.b4_ops[1 * i]
      h1 = op1(h1)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
      s = h1
      branch4_states += [s]

    branch1_output = torch.cat([branch1_states[i] for i in self._concat], dim=1)
    branch2_output = torch.cat([branch2_states[i] for i in self._concat], dim=1)
    branch3_output = torch.cat([branch3_states[i] for i in self._concat], dim=1)
    branch4_output = torch.cat([branch4_states[i] for i in self._concat], dim=1)
    branch1_output = self.branch1_att_preprocess(branch1_output)
    branch2_output = self.branch2_att_preprocess(branch2_output)
    branch3_output = self.branch3_att_preprocess(branch3_output)
    branch4_output = self.branch4_att_preprocess(branch4_output)

    output = torch.cat((branch1_output, branch2_output, branch3_output, branch4_output), dim=1)
    output = self.f_att_preprocess(output)
    output = branch5_fi + output

    return output

class AdjacentAggreCell (nn.Module): #ACSF steps=4 multiplier=4
  def __init__(self, genotype, index, steps, multiplier, C, parse_method):
    super(AdjacentAggreCell, self).__init__()
    op_names, indices = zip(*genotype.AdjacentAggre)
    concat = genotype.AdjacentAggre_concat

    if index == 0:
      self.fi_1_preprocess = nn.Sequential(
      )
    elif index == 1:
      self.fi_1_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
    elif index == 2:
      self.fi_1_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
    elif index == 3:
      self.fi_1_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
    elif index == 4:
      self.fi_1_preprocess = nn.Sequential(
      )

    self._steps = steps
    self._multiplier = multiplier
    self._compile(C, op_names, indices, concat)

    self.output_preprocess = nn.Sequential(
      nn.Conv2d(256, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True)
    )

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, fi, fi_1, drop_prob):
    fi_1 = self.fi_1_preprocess(fi_1)
    states = [fi, fi_1]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]

    output = torch.cat([states[i] for i in self._concat], dim=1)
    output = self.output_preprocess(output)
    output = output + fi
    return output

def get_dilated_mask(input, kernel_size, iterations):
  kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
  dilated_mask_list = map(
    lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
  dilated_mask_numpy = np.array(list(dilated_mask_list))
  dilated_mask_tensor = torch.from_numpy(dilated_mask_numpy)
  return dilated_mask_tensor.unsqueeze(1).cuda()

class LowAggreCell(nn.Module):  #EEIR1
  def __init__(self, genotype, steps, multiplier, aggbranch_steps, aggbranch_multiplier, C, parse_method):
    super(LowAggreCell, self).__init__()

    op_names_seg, indices_seg = zip(*genotype.Low_seg)
    concat_seg = genotype.Low_seg_concat
    op_names_edge, indices_edge = zip(*genotype.Low_edge)
    concat_edge = genotype.Low_edge_concat
    op_names_agg, indices_agg = zip(*genotype.Low_agg)
    concat_agg = genotype.Low_agg_concat

    self.p_preprocess = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    self.edge_preprocess = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    self.fp_preprocess = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    self.fedge_preprocess = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    self.fi_1_preprocess = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    self.f12_preprocess = nn.Sequential(
      nn.Conv2d(2 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True)
    )

    self.f12_edge_preprocess = nn.Sequential(
      nn.Conv2d(2 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True)
    )

    self.f1f2_preprocess = nn.Sequential(
      nn.Conv2d(2 * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True)
    )

    self._steps = steps
    self._multiplier = multiplier
    self._aggbranch_steps = aggbranch_steps
    self._aggbranch_multiplier = aggbranch_multiplier

    self._compile(C, op_names_seg, indices_seg, concat_seg, op_names_edge, indices_edge, concat_edge, op_names_agg, indices_agg, concat_agg)

    self.segbranch_low_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.edgebranch_low_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.aggbranch_low_preprocess = nn.Sequential(
      nn.Conv2d(self._aggbranch_multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.segbranch_out_preprocess = nn.Sequential(
      nn.Conv2d(2 * C, C, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.edgebranch_out_preprocess = nn.Sequential(
      nn.Conv2d(2 * C, C, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

  def _compile(self, C, op_names_seg, indices_seg, concat_seg, op_names_edge, indices_edge, concat_edge, op_names_agg, indices_agg, concat_agg):
    assert len(op_names_seg) == len(indices_seg)
    assert len(op_names_edge) == len(indices_edge)
    assert len(op_names_agg) == len(indices_agg)

    self._concat_seg = concat_seg
    self._concat_edge = concat_edge
    self._concat_agg = concat_agg
    self._ops_seg = nn.ModuleList()
    self._ops_edge = nn.ModuleList()
    self._ops_agg = nn.ModuleList()

    for name, index in zip(op_names_seg, indices_seg):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops_seg += [op]
    self._indices_seg = indices_seg

    for name, index in zip(op_names_edge, indices_edge):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops_edge += [op]
    self._indices_edge = indices_edge

    for name, index in zip(op_names_agg, indices_agg):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops_agg += [op]
    self._indices_agg = indices_agg

  def forward(self, f1, f2, f3, p3, edge3, fedge3, drop_prob):
    p3 = self.p_preprocess(p3)
    edge3 = self.edge_preprocess(edge3)
    f3 = self.fi_1_preprocess(f3)
    fedge3 = self.fedge_preprocess(fedge3)

    edge3 = torch.sigmoid(edge3)
    f1f2 = torch.cat((f1, f2), dim=1)
    f1f2 = self.f1f2_preprocess(f1f2)
    f1f2_edge_input = edge3.expand(-1, 64, -1, -1).mul(f1f2) + f1f2
    edgebranch_states = [f1f2_edge_input, f3]

    p3 = torch.sigmoid(p3)
    p3_exp = get_dilated_mask(p3, 5, 2)
    p3_edge = p3_exp - p3
    p3 = p3.expand(-1, 64, -1, -1)
    p3_edge = p3_edge.expand(-1, 64, -1, -1)
    f1_f = p3.mul(f1) + f1
    f2_f = p3.mul(f2) + f2
    f1_edge = p3_edge.mul(f1) + f1
    f2_edge = p3_edge.mul(f2) + f2
    f12 = torch.cat((f1_f, f2_f), 1)
    f12_edge = torch.cat((f1_edge, f2_edge), 1)
    f12 = self.f12_preprocess(f12)
    f12_edge = self.f12_edge_preprocess(f12_edge)
    segbranch_states = [f12, f12_edge, f3]

    for i in range(self._steps):
      h1 = segbranch_states[self._indices_seg[3 * i]]
      h2 = segbranch_states[self._indices_seg[3 * i + 1]]
      h3 = segbranch_states[self._indices_seg[3 * i + 2]]
      op1 = self._ops_seg[3 * i]
      op2 = self._ops_seg[3 * i + 1]
      op3 = self._ops_seg[3 * i + 2]
      h1 = op1(h1)
      h2 = op2(h2)
      h3 = op3(h3)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
        if not isinstance(op3, Identity):
          h3 = drop_path(h3, drop_prob)
      s = h1 + h2 + h3
      segbranch_states += [s]

      h1 = edgebranch_states[self._indices_edge[2 * i]]
      h2 = edgebranch_states[self._indices_edge[2 * i + 1]]
      op1 = self._ops_edge[2 * i]
      op2 = self._ops_edge[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      edgebranch_states += [s]

    segbranch_output = torch.cat([segbranch_states[i] for i in self._concat_seg], dim=1)
    edgebranch_output = torch.cat([edgebranch_states[i] for i in self._concat_edge], dim=1)

    segbranch_output = self.segbranch_low_preprocess(segbranch_output)
    edgebranch_output = self.edgebranch_low_preprocess(edgebranch_output)

    aggbranch_states = [segbranch_output, edgebranch_output]
    for i in range(self._aggbranch_steps):
      h1 = aggbranch_states[self._indices_agg[2 * i]]
      h2 = aggbranch_states[self._indices_agg[2 * i + 1]]
      op1 = self._ops_agg[2 * i]
      op2 = self._ops_agg[2 * i + 1]
      h1 = op1(h1)
      h2 = op1(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      aggbranch_states += [s]

    aggbranch_output = torch.cat([aggbranch_states[i] for i in self._concat_agg], dim=1)
    aggbranch_output = self.aggbranch_low_preprocess(aggbranch_output)

    edgebranch_output = torch.cat([aggbranch_output, edgebranch_output], dim=1)
    segbranch_output = torch.cat([aggbranch_output, segbranch_output], dim=1)

    edgebranch_output = self.edgebranch_out_preprocess(edgebranch_output) + fedge3
    segbranch_output = self.segbranch_out_preprocess(segbranch_output) + f3

    return segbranch_output, edgebranch_output


class RefineCell(nn.Module):  ##EEIR，steps=4，multiplier=4
  def __init__(self, genotype, index, steps, multiplier, aggbranch_steps, aggbranch_multiplier, C, parse_method):
    super(RefineCell, self).__init__()
    op_names_seg, indices_seg = zip(*genotype.Refine_seg)
    concat_seg = genotype.Refine_seg_concat
    op_names_edge, indices_edge = zip(*genotype.Refine_edge)
    concat_edge = genotype.Refine_edge_concat
    op_names_agg, indices_agg = zip(*genotype.Refine_agg)
    concat_agg = genotype.Refine_agg_concat

    self._steps = steps
    self._multiplier = multiplier
    self._aggbranch_steps = aggbranch_steps
    self._aggbranch_multiplier = aggbranch_multiplier

    self._compile(C, op_names_seg, indices_seg, concat_seg, op_names_edge, indices_edge, concat_edge, op_names_agg, indices_agg, concat_agg)

    if index == 0:
      self.fi_preprocess = nn.Sequential(
      )

      self.fi_1_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

      self.p_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

      self.edge_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

      self.fedge_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
    elif index == 1:
      self.fi_preprocess = nn.Sequential(
      )

      self.fi_1_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

      self.p_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

      self.edge_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

      self.fedge_preprocess = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    elif index == 2:
      self.fi_preprocess = nn.Sequential(
      )

      self.fi_1_preprocess = nn.Sequential(
      )

      self.p_preprocess = nn.Sequential(
      )

      self.edge_preprocess = nn.Sequential(
      )

      self.fedge_preprocess = nn.Sequential(
      )

    self.segbranch_ref_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.edgebranch_ref_preprocess = nn.Sequential(
      nn.Conv2d(self._multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.aggbranch_ref_preprocess = nn.Sequential(
      nn.Conv2d(self._aggbranch_multiplier * C, C, kernel_size=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

    self.segbranch_out_preprocess = nn.Sequential(
      nn.Conv2d(2 * C, C, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))
    self.edgebranch_out_preprocess = nn.Sequential(
      nn.Conv2d(2 * C, C, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(C, affine=True),
      nn.ReLU(inplace=True))

  def _compile(self, C, op_names_seg, indices_seg, concat_seg, op_names_edge, indices_edge, concat_edge, op_names_agg, indices_agg, concat_agg):
    assert len(op_names_seg) == len(indices_seg)
    assert len(op_names_edge) == len(indices_edge)
    assert len(op_names_agg) == len(indices_agg)

    self._concat_seg = concat_seg
    self._concat_edge = concat_edge
    self._concat_agg = concat_agg
    self._ops_seg = nn.ModuleList()
    self._ops_edge = nn.ModuleList()
    self._ops_agg = nn.ModuleList()

    for name, index in zip(op_names_seg, indices_seg):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops_seg += [op]
    self._indices_seg = indices_seg

    for name, index in zip(op_names_edge, indices_edge):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops_edge += [op]
    self._indices_edge = indices_edge

    for name, index in zip(op_names_agg, indices_agg):
      stride = 1
      op = OPS[name](C, stride, True)
      self._ops_agg += [op]
    self._indices_agg = indices_agg

  def forward(self, fi, fi_1, p, edge, fedge, drop_prob):
    fi = self.fi_preprocess(fi)
    fi_1 = self.fi_1_preprocess(fi_1)
    p = self.p_preprocess(p)
    edge = self.edge_preprocess(edge)
    fedge = self.fedge_preprocess(fedge)

    p = torch.sigmoid(p)
    p_r = -1 * p + 1
    edge = torch.sigmoid(edge)
    fi_edge_input = edge.expand(-1, 64, -1, -1).mul(fi) + fi
    edgebranch_states = [fi_edge_input, fi_1]

    fi_f = p.expand(-1, 64, -1, -1).mul(fi) + fi
    fi_r = p_r.expand(-1, 64, -1, -1).mul(fi) + fi
    segbranch_states = [fi_f, fi_r, fi_1]

    for i in range(self._steps):
      h1 = segbranch_states[self._indices_seg[3 * i]]
      h2 = segbranch_states[self._indices_seg[3 * i + 1]]
      h3 = segbranch_states[self._indices_seg[3 * i + 2]]
      op1 = self._ops_seg[3 * i]
      op2 = self._ops_seg[3 * i + 1]
      op3 = self._ops_seg[3 * i + 2]
      h1 = op1(h1)
      h2 = op2(h2)
      h3 = op3(h3)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
        if not isinstance(op3, Identity):
          h3 = drop_path(h3, drop_prob)
      s = h1 + h2 + h3
      segbranch_states += [s]

      h1 = edgebranch_states[self._indices_edge[2 * i]]
      h2 = edgebranch_states[self._indices_edge[2 * i + 1]]
      op1 = self._ops_edge[2 * i]
      op2 = self._ops_edge[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      edgebranch_states += [s]

    segbranch_output = torch.cat([segbranch_states[i] for i in self._concat_seg], dim=1)
    edgebranch_output = torch.cat([edgebranch_states[i] for i in self._concat_edge], dim=1)

    segbranch_output = self.segbranch_ref_preprocess(segbranch_output)
    edgebranch_output = self.edgebranch_ref_preprocess(edgebranch_output)

    aggbranch_states = [segbranch_output, edgebranch_output]
    for i in range(self._aggbranch_steps):
      h1 = aggbranch_states[self._indices_agg[2 * i]]
      h2 = aggbranch_states[self._indices_agg[2 * i + 1]]
      op1 = self._ops_agg[2 * i]
      op2 = self._ops_agg[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      aggbranch_states += [s]

    aggbranch_output = torch.cat([aggbranch_states[i] for i in self._concat_agg], dim=1)
    aggbranch_output = self.aggbranch_ref_preprocess(aggbranch_output)

    edgebranch_output = torch.cat([aggbranch_output, edgebranch_output], dim=1)
    segbranch_output = torch.cat([aggbranch_output, segbranch_output], dim=1)

    edgebranch_output = self.edgebranch_out_preprocess(edgebranch_output) + fedge
    segbranch_output = self.segbranch_out_preprocess(segbranch_output) + fi_1

    return segbranch_output, edgebranch_output

class NasCodNet(nn.Module):
  def __init__(self, channel=64, att_steps=2, att_multiplier=2, adj_steps=4, adj_multiplier=4, coa_steps=4, coa_multiplier=4,
               ref_steps = 4, ref_multiplier = 4, low_steps = 4, low_multiplier = 4, genotype=None, parse_method='darts'):
    super(NasCodNet, self).__init__()
    self.drop_path_prob = 0
    self.parse_method = parse_method

    self.resnet = res2net50_v1b_26w_4s(pretrained=True)

    self.Attention_1 = AttentionCell_1(genotype, att_steps, att_multiplier, 64, channel, parse_method)
    self.Attention_2 = AttentionCell_2(genotype, att_steps, att_multiplier, 256, channel, parse_method)
    self.Attention_3 = AttentionCell_3(genotype, att_steps, att_multiplier, 512, channel, parse_method)
    self.Attention_4 = AttentionCell_4(genotype, att_steps, att_multiplier, 1024, channel, parse_method)
    self.Attention_5 = AttentionCell_5(genotype, att_steps, att_multiplier, 2048, channel, parse_method)

    self.CoarseAggre_1 = CoarseAggreCell_1(genotype, coa_steps, coa_multiplier, channel, parse_method)
    self.CoarseAggre_2 = CoarseAggreCell_2(genotype, coa_steps, coa_multiplier, channel, parse_method)
    self.CoarseAdj = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.CoarseRefine = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    self.AdjacentAggre_1 = AdjacentAggreCell(genotype, 0, adj_steps, adj_multiplier, channel, parse_method)
    self.AdjacentAggre_2 = AdjacentAggreCell(genotype, 1, adj_steps, adj_multiplier, channel, parse_method)
    self.AdjacentAggre_3 = AdjacentAggreCell(genotype, 2, adj_steps, adj_multiplier, channel, parse_method)
    self.AdjacentAggre_4 = AdjacentAggreCell(genotype, 3, adj_steps, adj_multiplier, channel, parse_method)
    self.AdjacentAggre_5 = AdjacentAggreCell(genotype, 4, adj_steps, adj_multiplier, channel, parse_method)

    self.Refine_3 = RefineCell(genotype, 0, ref_steps, ref_multiplier, ref_steps, ref_multiplier, channel, parse_method)
    self.Refine_4 = RefineCell(genotype, 1, ref_steps, ref_multiplier, ref_steps, ref_multiplier, channel, parse_method)
    self.Refine_5 = RefineCell(genotype, 2, ref_steps, ref_multiplier, ref_steps, ref_multiplier, channel, parse_method)

    self.LowAggre = LowAggreCell(genotype, low_steps, low_multiplier, low_steps, low_multiplier, channel, parse_method)

    self.predict_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_coarse = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_coarse_edge = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_12 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_edge_3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_edge_4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_edge_5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    self.predict_edge_12 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
  def edge_enhance(self, img):
    bs, c, h, w = img.shape
    gradient = img.clone()
    gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
    gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
    out = img - gradient
    out = torch.clamp(out, 0, 1)
    return out

  def forward(self, x):
    x = self.resnet.conv1(x)
    x = self.resnet.bn1(x)
    x = self.resnet.relu(x)
    f1 = self.resnet.maxpool(x)
    f2 = self.resnet.layer1(f1)
    f3 = self.resnet.layer2(f2)
    f4 = self.resnet.layer3(f3)
    f5 = self.resnet.layer4(f4)

    att1 = self.Attention_1(f1, self.drop_path_prob)
    att2 = self.Attention_2(f2, self.drop_path_prob)
    att3 = self.Attention_3(f3, self.drop_path_prob)
    att4 = self.Attention_4(f4, self.drop_path_prob)
    att5 = self.Attention_5(f5, self.drop_path_prob)

    coarseP1 = self.CoarseAggre_1(att3, att4, self.drop_path_prob)
    coarseP2 = self.CoarseAggre_2(coarseP1, att5, self.drop_path_prob)

    coarse_adj = self.CoarseAdj(coarseP2)
    adj_agree_5 = self.AdjacentAggre_5(att5, coarse_adj, self.drop_path_prob)
    adj_agree_4 = self.AdjacentAggre_4(att4, att5, self.drop_path_prob)
    adj_agree_3 = self.AdjacentAggre_3(att3, att4, self.drop_path_prob)
    adj_agree_2 = self.AdjacentAggre_2(att2, att3, self.drop_path_prob)
    adj_agree_1 = self.AdjacentAggre_1(att1, att2, self.drop_path_prob)

    coarse_refine = self.CoarseRefine(coarseP2)
    coarse_map = self.predict_coarse(coarseP2)
    coarse_edge = self.predict_coarse_edge(coarseP2)

    refine_5, edge_5 = self.Refine_5(adj_agree_5, coarse_refine, coarse_map, coarse_edge, coarse_refine, self.drop_path_prob)
    map5 = self.predict_5(refine_5)
    edge5 = self.predict_edge_5(edge_5)
    refine_4, edge_4 = self.Refine_4(adj_agree_4, refine_5, map5, edge5, edge_5, self.drop_path_prob)
    map4 = self.predict_4(refine_4)
    edge4 = self.predict_edge_4(edge_4)
    refine_3, edge_3 = self.Refine_3(adj_agree_3, refine_4, map4, edge4, edge_4, self.drop_path_prob)
    map3 = self.predict_3(refine_3)
    edge3 = self.predict_edge_3(edge_3)

    refine_12, edge_12 = self.LowAggre(adj_agree_1, adj_agree_2, refine_3, map3, edge3, edge_3, self.drop_path_prob)
    map12 = self.predict_12(refine_12)
    edge12 = self.predict_edge_12(edge_12)

    out_coarse = F.interpolate(coarse_map, scale_factor=32, mode='bilinear')
    coarse_edge = self.edge_enhance(coarse_edge)
    out_edge_coarse = F.interpolate(coarse_edge, scale_factor=32, mode='bilinear')
    map5 = map5 + coarse_map
    out_5 = F.interpolate(map5, scale_factor=32, mode='bilinear')
    edge5 = self.edge_enhance(edge5)
    out_edge_5 = F.interpolate(edge5, scale_factor=32, mode='bilinear')
    out_4 = F.interpolate(map4, scale_factor=16, mode='bilinear')
    edge4 = self.edge_enhance(edge4)
    out_edge_4 = F.interpolate(edge4, scale_factor=16, mode='bilinear')
    out_3 = F.interpolate(map3, scale_factor=8, mode='bilinear')
    edge3 = self.edge_enhance(edge3)
    out_edge_3 = F.interpolate(edge3, scale_factor=8, mode='bilinear')
    out_12 = F.interpolate(map12, scale_factor=4, mode='bilinear')
    edge12 = self.edge_enhance(edge12)
    out_edge_12 = F.interpolate(edge12, scale_factor=4, mode='bilinear')

    return out_coarse, out_5, out_4, out_3, out_12, out_edge_coarse, out_edge_5, out_edge_4, out_edge_3, out_edge_12