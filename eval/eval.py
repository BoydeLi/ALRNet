import os
import sys
import cv2
from tqdm import tqdm
import metrics
import json
import argparse
import numpy as np


def eval(parser, dataset):
    args = parser.parse_args()

    FM = metrics.Fmeasure_and_FNR()
    WFM = metrics.WeightedFmeasure()
    SM = metrics.Smeasure()
    EM = metrics.Emeasure()
    MAE = metrics.MAE()

    model = args.model
    gt_root = args.GT_root
    pred_root = args.pred_root

    gt_root = os.path.join(gt_root, dataset)
    gt_root = os.path.join(gt_root, 'gt')
    pred_root = os.path.join(pred_root, dataset)

    gt_name_list = sorted(os.listdir(pred_root))

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)
        print(gt_path)
        print(pred_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt.shape != pred.shape:
            cv2.imwrite(os.path.join(pred_root, gt_name), cv2.resize(pred, gt.shape[::-1]))
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)

    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]
    model_r = str(args.model)
    Smeasure_r = str(sm.round(3)) ##保留三位小数
    Wmeasure_r = str(wfm.round(3))
    MAE_r = str(mae.round(3))
    adpEm_r = str(em['adp'].round(3))
    meanEm_r = str('-' if em['curve'] is None else em['curve'].mean().round(3))
    maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(3))
    adpFm_r = str(fm['adp'].round(3))
    meanFm_r = str(fm['curve'].mean().round(3))
    maxFm_r = str(fm['curve'].max().round(3))
    fnr_r = str(fnr.round(3))

    eval_record = str(
        'Model:' + model_r + ',' +
        'Dataset:' + dataset + '||' +
        'Smeasure:' + Smeasure_r + '; ' +
        'wFmeasure:' + Wmeasure_r + '; ' +
        'MAE:' + MAE_r + '; ' +
        'fnr:' + fnr_r + ';' +
        'adpEm:' + adpEm_r + '; ' +
        'meanEm:' + meanEm_r + '; ' +
        'maxEm:' + maxEm_r + '; ' +
        'adpFm:' + adpFm_r + '; ' +
        'meanFm:' + meanFm_r + '; ' +
        'maxFm:' + maxFm_r
    )

    print(eval_record)
    print('#' * 50)
    if args.record_path is not None:
        txt = args.record_path
    else:
        txt = 'output/eval_record.txt'
    f = open(txt, 'a')
    f.write(eval_record)
    f.write("\n")
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='NasCod')
    parser.add_argument("--pred_root", default='')
    parser.add_argument("--GT_root", default='.')
    parser.add_argument("--record_path", default=None)

    args = parser.parse_args()
    # datasets = ['NC4K', 'COD10K', 'CAMO', 'CHAMELEON']
    datasets = ['CAMO']
    existed_pred = os.listdir(args.pred_root)
    for dataset in datasets:
        if dataset in existed_pred:
            eval(parser, dataset)