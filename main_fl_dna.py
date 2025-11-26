import copy
import numpy as np
import os
import logging
import pathlib
import argparse
import time
import sys
import random
import math
from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
from transformers import get_cosine_schedule_with_warmup

from data.DNA_data import MyDataset, CustomSampler, CustomBatchSampler, collater
from models.Model import Encoder, Model
from models.val import val
from models.test import test_s
from utils.Loss import CEBayesRiskLoss, KLDivergenceLoss, SSBayesRiskLoss
from utils.uncertainty import scoring_func
from utils.Fed import FedAvg
from utils.utils import ema_update
from utils.weight import weight




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning for DNA")

    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--num_users', type=int, default=4, help="Number of users")
    parser.add_argument('--local_ep', type=int, default=2, help="Local epochs")
    parser.add_argument('--local_bs', type=int, default=32, help="Local batch size")
    parser.add_argument('--train_datasets', type=str, required=True, help="Training dataset")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--deterministic', type=bool, default=False, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dim', type=int, default=256, help='dim')

    args = parser.parse_args()

    path_dict = {'I': pathlib.Path('Dataset/I'),
                 'B': pathlib.Path('Dataset/B'),
                 'P': pathlib.Path('Dataset/P'),
                 'S': pathlib.Path('Dataset/S')
                 }
    torch.cuda.empty_cache()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(args.device)

    localtime = time.localtime(time.time())
    ticks = '{:>02d}{:>02d}{:>02d}{:>02d}{:>02d}'.format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                                         localtime.tm_min, localtime.tm_sec)

    snapshot_path = "result/FLDNA_{}/{}_{}/".format(args.train_datasets, args.train_datasets, ticks)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loaders = []
    val_loaders = []
    test_loaders = []
    train_num = []

    train_datasets_list = args.train_datasets.split(',')

    padding_length_dict = {'I': 155, 'B': 205, 'P': 188, 'S': 201}
    label_length_dict = {'I': 150, 'B': 200, 'P': 183, 'S': 196}

    for client_id in range(args.num_users):

        dataset_name = train_datasets_list[client_id]
        if dataset_name not in padding_length_dict:
            raise ValueError(f"Unknown dataset {dataset_name} in padding_length_dict.")

        train_collate_fn = collater(padding_length_dict[dataset_name])

        train_set = MyDataset(path_dict, datasets=[dataset_name], mode='train')
        train_num.append(len(train_set))
        train_cs = CustomSampler(data=train_set)
        train_bs = CustomBatchSampler(sampler=train_cs, batch_size=args.local_bs, drop_last=True)
        train_dl = DataLoader(dataset=train_set, batch_sampler=train_bs, collate_fn=train_collate_fn,
                              pin_memory=True, num_workers=8)
        train_loaders.append(train_dl)

        val_set = MyDataset(path_dict, datasets=[dataset_name], mode='val')
        val_cs = CustomSampler(data=val_set)
        val_bs = CustomBatchSampler(sampler=val_cs, batch_size=args.local_bs, drop_last=True)
        val_dl = DataLoader(dataset=val_set, batch_sampler=val_bs, collate_fn=train_collate_fn,
                            pin_memory=True, num_workers=8)
        val_loaders.append(val_dl)

        test_set = MyDataset(path_dict, datasets=[dataset_name], mode='test')
        test_cs = CustomSampler(data=test_set)
        test_bs = CustomBatchSampler(sampler=test_cs, batch_size=args.local_bs, drop_last=True)
        test_dl = DataLoader(dataset=test_set, batch_sampler=test_bs, collate_fn=train_collate_fn,
                             pin_memory=True, num_workers=8)
        test_loaders.append(test_dl)

    for client_id, train_loader in enumerate(train_loaders):
        print(f"Client {client_id + 1} - Number of training samples: {len(train_loader.dataset)}")

    writer = SummaryWriter(snapshot_path + '/log')

    result_after_avg = np.zeros(args.num_users)

    best_val = 9999

    global_encoder = Encoder(
        dim=args.dim
    ).to(args.device)

    encoders = []
    models = []
    for client_id in range(args.num_users):
        dataset_name = train_datasets_list[client_id]
        encoder = Encoder(
            dim=args.dim
        ).to(args.device)
        encoders.append(encoder)

        model = Model(encoder, args.dim, padding_length_dict[dataset_name], label_length_dict[dataset_name]).to(
            args.device)
        models.append(model)

    optimizers = []
    schedulers = []
    for client_id in range(args.num_users):
        optimizer = torch.optim.AdamW(models[client_id].parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-09, weight_decay=0,
                                      amsgrad=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0, last_epoch=-1)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    client_weight = train_num / np.sum(train_num)

    with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
        print('train num: {}'.format(args.num_users), file=f)
        print('init weight: {}'.format(client_weight), file=f)

    bayes_risk = CEBayesRiskLoss().to(args.device)
    kld_loss = KLDivergenceLoss().to(args.device)
    u_gepi = np.zeros((args.num_users, args.epochs))
    u_lepi = np.zeros((args.num_users, args.epochs))
    u_gale = np.zeros((args.num_users, args.epochs))
    u_lale = np.zeros((args.num_users, args.epochs))

    start_epoch = -1
    for epoch in range(start_epoch + 1, args.epochs):
        for idx in range(args.num_users):
            dataset_name = train_datasets_list[idx]
            model = models[idx]
            model.train()
            train_dl = train_loaders[idx]

            optimizer = optimizers[idx]
            scheduler = schedulers[idx]

            size = len(train_dl.dataset)

            epoch_loss = 0
            num_batches = 0

            for t in range(args.local_ep):
                current = 0
                for i, data in enumerate(train_dl):
                    inputs, labels = data
                    inputs, labels = Variable(inputs.float()).to(args.device), Variable(labels).to(
                        args.device)
                    evidences = model(inputs)
                    eye = torch.eye(4, dtype=torch.float32, device=args.device)
                    labels = eye[labels]
                    annealing_coef = min(1.0, epoch / args.epochs)
                    loss = bayes_risk(evidences, labels) + annealing_coef * kld_loss(evidences, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    current += inputs.size(0)
                    epoch_loss += loss.item()
                    num_batches += 1
                    if (i+1) % 50 == 0:
                        print(
                            f'Global Epoch: {epoch + 1} Local Epoch: {t + 1} Client {idx + 1}:{train_datasets_list[idx]} \n'
                            f'Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n'
                            f'----------------------')
                avg_loss = epoch_loss / num_batches
                writer.add_scalar(f'{train_datasets_list[idx]}_train_loss', avg_loss, epoch * args.local_ep + t)

            u_gepi[idx, epoch], u_lepi[idx, epoch], u_gale[idx, epoch], u_lale[idx, epoch] = scoring_func(global_encoder, model,
                                                                                     train_loaders[idx], padding_length_dict[dataset_name], label_length_dict[dataset_name], args)
            with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                print('client {}:\tu_gepi\t{:.5f}\tu_lepi\t{:.5f}\tu_gale\t{:.5f}\tu_lale\t{:.5f}'.format(
                    train_datasets_list[idx],
                    u_gepi[idx, epoch], u_lepi[idx, epoch],
                    u_gale[idx, epoch], u_lale[idx, epoch]
                ), file=f)

        client_weight = weight(client_weight, u_gepi, u_gale, u_lepi, u_lale, epoch)
        global_encoder = FedAvg(global_encoder, models, client_weight)

        for idx in range(args.num_users):
            models[idx].encoder.load_state_dict(global_encoder.state_dict())
            result_after_avg[idx] = val(model=models[idx], dataloader=val_loaders[idx], epoch=epoch, args=args)
            writer.add_scalar(f'{train_datasets_list[idx]}_val_loss', result_after_avg[idx], epoch)

        avg_val = result_after_avg.mean()
        print("avg_val:",avg_val)
        writer.add_scalar('average_val_loss', avg_val, epoch)

        if avg_val < best_val:
            best_val = avg_val

            with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                print('FL round {}'.format(epoch + 1), file=f)
                print('weight: {}'.format(client_weight), file=f)

            for idx in range(args.num_users):
                acc = test_s(model=models[idx], dataloader=test_loaders[idx], args=args)
                writer.add_scalar(f'{train_datasets_list[idx]}_acc', acc, epoch)
                print('client {}: acc {:.5f}'.format(train_datasets_list[idx], acc))
                with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                    print('client {}.\tacc\t{:.5f}'.format(train_datasets_list[idx], acc),
                          file=f)
                save_model_path = os.path.join(
                    snapshot_path + 'model/epoch{}_{}.pth'.format(epoch+1, train_datasets_list[idx]))
                torch.save(models[idx].state_dict(), save_model_path)

    writer.close()



