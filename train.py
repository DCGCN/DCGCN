from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

from lib import utils
from lib.utils import log_string, loadData, loadDatatime
from model.DSTGNN import DSTGNN

parser = argparse.ArgumentParser()
# parser.add_argument('--time_slot', type = int, default = 5,
#                     help = 'a time step is 5 mins')
parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 12,
                    help = 'prediction steps')
parser.add_argument('--L', type = int, default = 5,
                    help = 'number of STAtt Blocks')
parser.add_argument('--K', type = int, default = 8,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 8,
                    help = 'dims of each head attention outputs')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 16,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 8,
                    help = 'epoch to run')
# parser.add_argument('--patience', type = int, default = 10,
#                     help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--traffic_file', default = 'data/METR-LA/metr-la.h5',
                    help = 'traffic file')
parser.add_argument('--SE_file', default = 'data/METR-LA/SE(METR).txt',
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = 'data/METR-LA/METR',
                    help = 'save the model to disk')
parser.add_argument('--log_file', default = 'data/METR-LA/log(METR)',
                    help = 'log file')
args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")

# trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = loadData(args)
trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = loadDatatime(args)




# adj = np.load('./data/metr_adj.npy')

log_string(log, "loading end....")

def res(model, valX, valTE, valY, mean, std):
    model.eval() # 评估模式, 这会关闭dropout
    # it = test_iter.get_iterator()
    num_val = valX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat = model(X, te)

                pred.append(y_hat.cpu().numpy()*std+mean)
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    # print(pred.shape, label.shape)
    maes = []
    rmses = []
    mapes = []

    for i in range(12):
        mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        # if i == 11:
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
            # print('step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
    
    mae, rmse, mape = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
    # print('average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
    
    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

def train(model, trainX, trainTE, trainY, valX, valTE, valY, mean, std):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 6, 7, 8],
                                                            gamma=0.2)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,    
    #                                 verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        # model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(trainX[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)

                optimizer.zero_grad()

                y_hat = model(X, te)

                # rate = 1 - ((epoch - 1) / 100) ** 2
                # loss = (1-rate) * _compute_loss(y_hat, y, True) + rate * _compute_loss(y_hat, y, False)
                loss = _compute_loss(y, y_hat*std+mean)
                # print(loss)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                # print(f"\nbatch loss: {l.cpu().item()}")
                n += y.shape[0]
                batch_count += 1
                pbar.update(1)
        # lr = lr_scheduler.get_lr()
        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        # print('epoch %d, lr %.6f, loss %.4f, time %.1f sec'
        #       % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
        mae, rmse, mape = res(model, valX, valTE, valY, mean, std)
        lr_scheduler.step()
        # lr_scheduler.step(mae)
        if mae[-1] < min_loss:
            min_loss = mae[-1]
            torch.save(model, args.model_file)

def test(model, valX, valTE, valY, mean, std):
    model = torch.load(args.model_file)
    mae, rmse, mape = res(model, valX, valTE, valY, mean, std)
    return mae, rmse, mape

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_loss(y_pred, y_true, flag):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    if flag == True:
        loss = loss * mask_l
    return loss.mean()

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

if __name__ == '__main__':
    maes, rmses, mapes = [], [], []
    for i in range(10):
        log_string(log, "model constructed begin....")
        model = DSTGNN(SE, 1, args.K*args.d, args.K, args.d, args.L).to(device)
        log_string(log, "model constructed end....")
        log_string(log, "train begin....")
        train(model, trainX, trainTE, trainY, testX, testTE, testY, mean, std)
        log_string(log, "train end....")
        mae, rmse, mape = test(model, testX, testTE, testY, mean, std)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
    log_string(log, "\n\nresults:")
    maes = np.stack(maes, 1)
    rmses = np.stack(rmses, 1)
    mapes = np.stack(mapes, 1)
    for i in range(12):
        log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (i+1, maes[i].mean(), rmses[i].mean(), mapes[i].mean()))
        log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (i+1, maes[i].std(), rmses[i].std(), mapes[i].std()))
    log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f' % (maes[-1].mean(), rmses[-1].mean(), mapes[-1].mean()))
    log_string(log, 'average, mae %.4f, rmse %.4f, mape %.4f' % (maes[-1].std(), rmses[-1].std(), mapes[-1].std()))
