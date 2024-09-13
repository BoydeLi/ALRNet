import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


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

def get_coef(iter_percentage, method):
  if method == "linear":
    milestones = (0.3, 0.7)
    coef_range = (0, 1)
    min_point, max_point = min(milestones), max(milestones)
    min_coef, max_coef = min(coef_range), max(coef_range)
    if iter_percentage < min_point:
      ual_coef = min_coef
    elif iter_percentage > max_point:
      ual_coef = max_coef
    else:
      ratio = (max_coef - min_coef) / (max_point - min_point)
      ual_coef = ratio * (iter_percentage - min_point)
  elif method == "cos":
    coef_range = (0, 1)
    min_coef, max_coef = min(coef_range), max(coef_range)
    normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
    ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
  else:
    ual_coef = 1.0
  return ual_coef


def cal_ual(seg_logits, seg_gts):
  assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
  sigmoid_x = seg_logits.sigmoid()
  loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
  return loss_map.mean()


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def save_scripts(path, scripts_to_save=None):
  if scripts_to_save is not None:
    script_path = os.path.join(path, 'scripts')
    if not os.path.exists(script_path):
      os.mkdir(script_path)
    for script in scripts_to_save:
      dst_file = os.path.join(script_path, os.path.basename(script))
      shutil.copyfile(script, dst_file)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
  decay = decay_rate ** (epoch // decay_epoch)
  for param_group in optimizer.param_groups:
    param_group['lr'] = decay * init_lr
    lr = param_group['lr']
  return lr