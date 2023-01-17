'''
Date: 2022-02-23 03:09:35
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-03-01 06:18:03
'''
# Reference: https://github.com/CVMI-Lab/PAConv/blob/main/obj_cls/eval_voting.py

from __future__ import print_function
import os
import argparse
import hydra
import omegaconf
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_util import ModelNet40, ScanObjectNN
from models.PointConT import PointConT_cls
import numpy as np
import rsmix_provider
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

def voting_test(args):
    logger = logging.getLogger(__name__)
    logger.info('Working path: %s' % str(os.getcwd()))
    logger.info('random seed is set to %s ...' % str(args.seed))

    # data loading
    logger.info('Load %s dataset ...' % args.dataset)
    DATA_PATH = hydra.utils.to_absolute_path(args.dataset_dir)

    if args.dataset == 'ModelNet40':
        test_loader = DataLoader(ModelNet40(DATA_PATH, partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)                                      
    elif args.dataset == 'ScanObjectNN':
        test_loader = DataLoader(ScanObjectNN(DATA_PATH, partition='test', num_points=args.num_points), num_workers=8,
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise NotImplementedError 

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger.info('Using GPUs : %s' % str(args.gpu))


    # model loading
    logger.info('Load %s model ...' % args.model_name)
    model = PointConT_cls(args).cuda()
    # model = nn.DataParallel(model)

    logger.info('Loading pretrained model...')
    checkpoint = torch.load('model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()
    best_acc = 0

    for i in range(args.num_repeat):
        test_true = []
        test_pred = []

        for data, label in tqdm(test_loader):           
            pred = 0
            for v in range(args.num_vote):
                new_data = data  
                new_label = label         
                r = np.random.rand(1)
                if args.beta > 0 and r < args.rsmix_prob:
                    new_data = new_data.cpu().numpy() 
                    new_data, lam, new_label, label_b = rsmix_provider.rsmix(
                        new_data, new_label, beta=args.beta, n_sample=args.rsmix_nsample)
                    new_data = torch.FloatTensor(new_data)
                new_data = new_data.cuda()
                with torch.no_grad():
                    pred += F.softmax(model(new_data), dim=1)
            pred /= args.num_vote
            label = label.cuda().squeeze()
            label = label.view(-1)
            pred_choice = pred.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        test_acc_avg = metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_avg = test_acc_avg
        logger.info('Voting %d, test acc: %.6f, test avg acc: %.6f,' % (i+1, test_acc*100, test_acc_avg*100))
        logger.info('best acc: %.6f, best avg acc: %.6f,' % (best_acc*100, best_acc_avg*100))

    logger.info('Final voting result test acc: %.6f, test avg acc: %.6f,' % (best_acc*100, best_acc_avg*100))


@hydra.main(config_path='config', config_name='voting_cls')
def main(args):
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    voting_test(args)


if __name__ == "__main__":
    main()
