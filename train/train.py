import os
import sys
sys.path.insert(0, sys.path[0]+"/../")
import numpy as np
import time
import torch
import glob
import logging
import argparse
import cv2
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from search import utils
from model.model import NasCodNet as Network
from data import dataloader_edge as dataloader
from search.utils import adjust_lr
from search import genotypes
from eval.metrics import Smeasure
from skimage import img_as_ubyte

parser = argparse.ArgumentParser("eval")
parser.add_argument('--data', type=str, default='../../COD_dataset/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=0.1, help='gradient clipping')
parser.add_argument('--decay_rate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=50, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
parser.add_argument('--att_steps', type=int, default=2, help='num of nodes in Attention Module')
parser.add_argument('--att_multiplier', type=int, default=2, help='num of concat nodes in Attention Module')
parser.add_argument('--adj_steps', type=int, default=4, help='num of nodes in Adjacent fusion Module')
parser.add_argument('--adj_multiplier', type=int, default=4, help='num of concat nodes in Adjacent fusion Module')
parser.add_argument('--coa_steps', type=int, default=4, help='num of nodes in Coarse Predict Module')
parser.add_argument('--coa_multiplier', type=int, default=4, help='num of concat nodes in Coarse Predict Module')
parser.add_argument('--ref_steps', type=int, default=4, help='num of nodes in Refine Module')
parser.add_argument('--ref_multiplier', type=int, default=4, help='num of concat nodes in Refine Module')
parser.add_argument('--low_steps', type=int, default=4, help='num of nodes in Low fusion Module')
parser.add_argument('--low_multiplier', type=int, default=4, help='num of concat nodes in Low fusion Module')
parser.add_argument('--adp_steps', type=int, default=4, help='num of nodes in Low fusion Module')
parser.add_argument('--adp_multiplier', type=int, default=4, help='num of concat nodes in Low fusion Module')

args = parser.parse_args()

def create_log():
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('../*.py'))
  utils.save_scripts(args.save, scripts_to_save=glob.glob('../search/*.py'))
  utils.save_scripts(args.save, scripts_to_save=glob.glob('../model/*.py'))
  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  return logging

def structure_loss(pred, mask):
  """
  loss function (ref: F3Net-AAAI-2020)
  """
  weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
  wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
  wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
  pred = torch.sigmoid(pred)
  inter = ((pred * mask) * weit).sum(dim=(2, 3))
  union = ((pred + mask) * weit).sum(dim=(2, 3))
  wiou = 1 - (inter + 1) / (union - inter + 1)
  return (wbce + wiou).mean()

def dice_loss(predict, target):
  smooth = 1
  p = 2
  valid_mask = torch.ones_like(target)
  predict = predict.contiguous().view(predict.shape[0], -1)
  target = target.contiguous().view(target.shape[0], -1)
  valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
  num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
  den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
  loss = 1 - num / den
  return loss.mean()

def main():
  args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  print(args.save)
  logging = create_log()

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, args.att_steps, args.att_multiplier, args.adj_steps, args.adj_multiplier,
                  args.coa_steps, args.coa_multiplier, args.ref_steps, args.ref_multiplier, args.low_steps,
                  args.low_multiplier, genotype)
  if args.parallel:
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
  else:
    model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
  train_root = os.path.join(args.data, 'train_set')
  test_root = os.path.join(args.data, 'test_set/COD10K')
  train_loader, trainset_num = dataloader.get_loader(image_root=train_root+'/img/', gt_root=train_root+'/gt/', edge_root=train_root+'/edge/', batchsize=args.batch_size, trainsize=416,
                                      shuffle=True, num_workers=2, pin_memory=True)
  test_loader, testset_num = dataloader.get_testloader(image_root=test_root + '/img/', gt_root=test_root + '/gt/', batchsize=1, testsize=416,
                                           shuffle=False, num_workers=2, pin_memory=True)
  writer = SummaryWriter(args.save)

  global best_epoch, best_sm_epoch, best_mae, best_sm
  reload_model = False
  if(reload_model):
    start_epo = 31
    para = torch.load('./eval-EXP-20240730-002519/Net_epoch_best_mae.pth')
    model.load_state_dict(para)
    best_mae = 1
    best_epoch = 0
    best_sm = 0
    best_sm_epoch = 0
    for epoch in range(start_epo+1,args.epochs):
      cur_lr = adjust_lr(optimizer, args.learning_rate, epoch, args.decay_rate, args.decay_period)
      logging.info('epoch %d lr %e', epoch, cur_lr)
      writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
      train_loss = train(train_loader, model, structure_loss, optimizer, writer, epoch, trainset_num, dice_loss)
      logging.info('train epoch %d train_loss %f', epoch, train_loss)
      mae, valid_loss, sm = infer(test_loader, model, writer, testset_num, structure_loss, epoch)
      logging.info('test epoch %f valid_loss %f valid_mae %f sm %f', epoch, valid_loss, mae, sm)
      if epoch == 0:
        best_mae = mae
        best_sm = sm
        best_epoch = epoch
        best_sm_epoch = epoch
        torch.save(model.state_dict(), args.save + '/Net_epoch_0' + '.pth')
      else:
        if sm > best_sm:
          best_sm = sm
          if not best_epoch == epoch:
            best_epoch = epoch
            torch.save(model.state_dict(), args.save + '/Net_epoch_best_' + str(epoch) + '.pth')
        if mae < best_mae:
          best_mae = mae
          if not best_epoch == epoch:
            best_epoch = epoch
            torch.save(model.state_dict(), args.save + '/Net_epoch_best_' + str(epoch) + '.pth')
      logging.info('[eval Info]:Epoch:{} MAE:{} bestMAE:{} SM:{} bestSM:{}'.format(epoch, mae, best_mae, sm, best_sm))
  else:
    for epoch in range(args.epochs):
      cur_lr = adjust_lr(optimizer, args.learning_rate, epoch, args.decay_rate, args.decay_period)
      logging.info('epoch %d lr %e', epoch, cur_lr)
      writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
      train_loss = train(train_loader, model, structure_loss, optimizer, writer, epoch, trainset_num, dice_loss)
      logging.info('train epoch %d train_loss %f', epoch, train_loss)
      mae, valid_loss, sm = infer(test_loader, model, writer, testset_num, structure_loss, epoch)
      logging.info('test epoch %f valid_loss %f valid_mae %f sm %f', epoch, valid_loss, mae, sm)
      if epoch == 0:
        best_mae = mae
        best_sm = sm
        best_epoch = epoch
        best_sm_epoch = epoch
        torch.save(model.state_dict(), args.save + '/Net_epoch_0' + '.pth')
      else:
        if sm > best_sm:
          best_sm = sm
          if not best_epoch == epoch:
            best_epoch = epoch
            torch.save(model.state_dict(), args.save + '/Net_epoch_best_' + str(epoch) + '.pth')
        if mae < best_mae:
          best_mae = mae
          if not best_epoch == epoch:
            best_epoch = epoch
            torch.save(model.state_dict(), args.save + '/Net_epoch_best_' + str(epoch) + '.pth')
      logging.info('[eval Info]:Epoch:{} MAE:{} bestMAE:{} SM:{} bestSM:{}'.format(epoch, mae, best_mae, sm, best_sm))



def train(train_queue, model, criterion, optimizer, writer, epo, trainset_num, edge_criterion):
  loss_record = utils.AvgrageMeter()
  loss5_record = utils.AvgrageMeter()
  loss4_record = utils.AvgrageMeter()
  loss3_record = utils.AvgrageMeter()
  loss12_record = utils.AvgrageMeter()
  losscoarse_record = utils.AvgrageMeter()
  model.train()
  for step, (input, target, name, edges) in enumerate(train_queue):
    target = target.cuda()
    input = input.cuda()
    edges = edges.cuda()
    n = input.size(0)
    optimizer.zero_grad()
    out_coarse, out_5, out_4, out_3, out_12, out_edge_coarse, out_edge_5, out_edge_4, out_edge_3, out_edge_12 = model(input)
    loss_coarse = criterion(out_coarse, target)
    loss_5 = criterion(out_5, target)
    loss_4 = criterion(out_4, target)
    loss_3 = criterion(out_3, target)
    loss_12 = criterion(out_12, target)
    loss = loss_coarse + loss_5 + loss_4 + loss_3 + loss_12
    edge_loss_coarse = edge_criterion(out_edge_coarse, edges)
    edge_loss_5 = edge_criterion(out_edge_5, edges)
    edge_loss_4 = edge_criterion(out_edge_4, edges)
    edge_loss_3 = edge_criterion(out_edge_3, edges)
    edge_loss_12 = edge_criterion(out_edge_12, edges)
    edge_loss = edge_loss_coarse + edge_loss_5 + edge_loss_4 + edge_loss_3 + edge_loss_12
    loss = loss + edge_loss

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    loss_record.update(loss.item(), n)
    losscoarse_record.update(loss_coarse.item(), n)
    loss5_record.update(loss_5.item(), n)
    loss4_record.update(loss_4.item(), n)
    loss3_record.update(loss_3.item(), n)
    loss12_record.update(loss_12.item(), n)

    if step % 5 == 0:
      global_step = epo * trainset_num + (step + 1) * n
      writer.add_scalar('searching train loss', loss, global_step)
      writer.add_scalar('searching train loss_coarse', loss_coarse, global_step)
      writer.add_scalar('searching train loss_5', loss_5, global_step)
      writer.add_scalar('searching train loss_4', loss_4, global_step)
      writer.add_scalar('searching train loss_3', loss_3, global_step)
      writer.add_scalar('searching train loss_12', loss_12, global_step)

    if step % args.report_freq == 0:
      logging.info('train step: %03d loss: %e', step, loss_record.avg)
  return loss_record.avg



def infer(valid_queue, model, writer, testset_num, criterion, epo):
  loss_record = utils.AvgrageMeter()
  loss5_record = utils.AvgrageMeter()
  loss4_record = utils.AvgrageMeter()
  loss3_record = utils.AvgrageMeter()
  loss12_record = utils.AvgrageMeter()
  losscoarse_record = utils.AvgrageMeter()
  model.eval()
  mae_sum = 0
  mae = 0
  mae_coa = 0
  mae_5 = 0
  mae_4 = 0
  mae_3 = 0
  SM = Smeasure()
  with torch.no_grad():
    for step, (input, target, name) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()
      out_coarse, out_5, out_4, out_3, out_12, out_edge_coarse, out_edge_5, out_edge_4, out_edge_3, out_edge_12 = model(
        input)
      loss_coarse = criterion(out_coarse, target)
      loss_5 = criterion(out_5, target)
      loss_4 = criterion(out_4, target)
      loss_3 = criterion(out_3, target)
      loss_12 = criterion(out_12, target)
      loss = loss_coarse + loss_5 + loss_4 + loss_3 + loss_12

      batch_len = input.size(0)
      loss_record.update(loss.item(), batch_len)
      losscoarse_record.update(loss_coarse.item(), batch_len)
      loss5_record.update(loss_5.item(), batch_len)
      loss4_record.update(loss_4.item(), batch_len)
      loss3_record.update(loss_3.item(), batch_len)
      loss12_record.update(loss_12.item(), batch_len)

      target = np.asarray(target.data.cpu().numpy(), np.float32)

      for i in range(batch_len):
        res = out_12[i]
        res = res.sigmoid().data.cpu().numpy().squeeze()
        gt = target[i][0]
        SM.step_by_train(pred=img_as_ubyte(res), gt_name=name[i])
        gt /= (gt.max() + 1e-8)
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
      if step % 5 == 0:
        global_step = epo * testset_num + (step + 1) * batch_len
        writer.add_scalar('eval eval loss', loss, global_step)
        writer.add_scalar('eval eval loss_coarse', loss_coarse, global_step)
        writer.add_scalar('eval eval loss_5', loss_5, global_step)
        writer.add_scalar('eval eval loss_4', loss_4, global_step)
        writer.add_scalar('eval eval loss_3', loss_3, global_step)
        writer.add_scalar('eval eval loss_12', loss_12, global_step)

  sm = SM.get_results()['sm']
  mae = mae_sum / testset_num
  return mae, loss_record.avg, sm

if __name__ == '__main__':
  main() 
