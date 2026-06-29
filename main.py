import argparse 
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm

import kitti_dataset
import nclt_dataset 
import REIN as rein_module
from REIN import REIN, StudentREIN

def get_args():
    parser = argparse.ArgumentParser(description='BEVPlace++ 知識蒸餾（治本修正版）')
    parser.add_argument('--batchSize', type=int, default=4, help='Number of triplets')
    parser.add_argument('--cacheBatchSize', type=int, default=8, help='Batch size for caching and testing')
    parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--runsPath', type=str, default='./runs_distill/', help='Path to save runs to.')
    parser.add_argument('--cachePath', type=str, default='./cache_distill/', help='Path to save cache to.')
    parser.add_argument('--teacher_path', type=str, default='./runs/Aug08_10-17-29/model_best.pth.tar', help='老師模型權重路徑 (.pth.tar)')
    parser.add_argument('--kitti_path', type=str, default='./datasets/KITTI/', help='KITTI 資料集根目錄')
    parser.add_argument('--nclt_path', type=str, default='./datasets/NCLT/', help='NCLT 資料集根目錄')
    
    # 數學對齊後，超參數回歸正常合理的 1.0 範圍
    parser.add_argument('--alpha', type=float, default=1.0, help='Loss_Global 權重')
    parser.add_argument('--beta', type=float, default=1.0, help='Loss_Spatial 權重')
    
    opt = parser.parse_args()
    return opt

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3
    def forward(self, anchor, positive, negative):
        if anchor.dim() == 1:
            pos_dist = torch.norm(anchor - positive, p=2)
            neg_dist = torch.norm(anchor.unsqueeze(0) - negative, dim=1, p=2)
        else:
            pos_dist = torch.norm(anchor - positive, dim=-1, p=2)
            neg_dist = torch.norm(anchor.unsqueeze(1) - negative, dim=-1, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss

def train_epoch_distill(epoch, teacher_model, student_model, train_set, optimizer, criterion, distill_criterion, writer, device, opt):
    epoch_loss = 0
    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize
    
    # ====== 1. Building Cache (🔥 核心對齊：強迫用 99% 的老師來幫學生做 Hard Mining 篩選) ======
    if epoch >= 0:
        print('====> Building Cache for Hard Mining (Using 99% Teacher Model)')
        train_set.mining = False
        train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
        
        if not exists(opt.cachePath):
            makedirs(opt.cachePath)
            
        tmp_cache = train_set.cache + '.tmp'
        with h5py.File(tmp_cache, mode='w') as h5: 
            # 注意：這裡的 pool_size 必須是老師模型的輸出維度（也是學生的 global_feat_dim，兩者均為 8192）
            pool_size = teacher_model.global_feat_dim  
            h5feat = h5.create_dataset("features", [len(train_set), pool_size], dtype=np.float32)
            
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                             batch_size=opt.cacheBatchSize, shuffle=False, 
                                             collate_fn=kitti_dataset.collate_fn)
            
            # 凍結老師進行高質量快取提取
            teacher_model.eval()
            with torch.no_grad():
                for iteration, (query, _, _, indices) in enumerate(training_data_loader, 1):
                    query = query.to(device)
                    _, _, global_descs = teacher_model(query)
                    h5feat[indices, :] = global_descs.detach().cpu().numpy()

        try:
            os.replace(tmp_cache, train_set.cache)
        except Exception:
            try:
                os.rename(tmp_cache, train_set.cache)
            except Exception:
                raise

        train_set.mining = True
        train_set.refreshCache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ====== 2. 蒸餾訓練主循環 ======
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                     batch_size=opt.batchSize, shuffle=True, 
                                     collate_fn=kitti_dataset.collate_fn)
    
    teacher_model.eval() 
    student_model.train()

    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader):
        B = query.shape[0]
        num_negs = negatives.shape[0] // B

        query = query.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        optimizer.zero_grad()

        # 老師同步提取完整三元組特徵
        with torch.no_grad():
            _, t_local_Q, t_global_Q = teacher_model(query)
            _, t_local_P, t_global_P = teacher_model(positives)
            _, t_local_N, t_global_N = teacher_model(negatives)

        # 學生提取對應特徵
        _, s_local_Q, s_global_Q = student_model(query)
        _, s_local_P, s_global_P = student_model(positives)
        _, s_local_N, s_global_N = student_model(negatives)

        # --- 損失函數計算 ---
        
        # (1) Loss_Triplet (學生自學，維護基本邊界)
        loss_triplet = torch.tensor(0.0, device=device)
        for i in range(B):
            max_loss = torch.max(criterion(
                s_global_Q[i], 
                s_global_P[i], 
                s_global_N[num_negs * i : num_negs * (i + 1)]
            ))
            loss_triplet += max_loss
        loss_triplet /= B
        
        # (2) Loss_Global (全局特徵蒸餾 - 採用 sum 並除以實際總元素量，治本對齊數量級)
        total_samples = B + B + (B * num_negs)
        loss_global_sum = distill_criterion(s_global_Q, t_global_Q) + \
                           distill_criterion(s_global_P, t_global_P) + \
                           distill_criterion(s_global_N, t_global_N)
        loss_global = loss_global_sum / total_samples
        
        # (3) Loss_Spatial (空間特徵蒸餾 - 壓縮通道維度，避免規格不對稱)
        t_spatial_Q = t_local_Q.mean(dim=1, keepdim=True)
        t_spatial_P = t_local_P.mean(dim=1, keepdim=True)
        t_spatial_N = t_local_N.mean(dim=1, keepdim=True)

        s_spatial_Q = s_local_Q.mean(dim=1, keepdim=True)
        s_spatial_P = s_local_P.mean(dim=1, keepdim=True)
        s_spatial_N = s_local_N.mean(dim=1, keepdim=True)

        if s_spatial_Q.shape[2:] != t_spatial_Q.shape[2:]:
            s_spatial_Q = F.interpolate(s_spatial_Q, size=t_spatial_Q.shape[2:], mode='bilinear', align_corners=False)
            s_spatial_P = F.interpolate(s_spatial_P, size=t_spatial_P.shape[2:], mode='bilinear', align_corners=False)
            s_spatial_N = F.interpolate(s_spatial_N, size=t_spatial_N.shape[2:], mode='bilinear', align_corners=False)

        spatial_pixels = t_spatial_Q.shape[2] * t_spatial_Q.shape[3]
        total_spatial_elements = total_samples * spatial_pixels
        
        loss_spatial_sum = distill_criterion(s_spatial_Q, t_spatial_Q) + \
                            distill_criterion(s_spatial_P, t_spatial_P) + \
                            distill_criterion(s_spatial_N, t_spatial_N)
        loss_spatial = loss_spatial_sum / total_spatial_elements
        
        # 總損失結合
        loss_total = loss_triplet + (opt.alpha * loss_global) + (opt.beta * loss_spatial)

        # 反向傳播與更新
        loss_total.backward()
        optimizer.step()

        batch_loss = loss_total.item()
        epoch_loss += batch_loss
        
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f} | Triplet: {:.4f} | Global_KD: {:.4f} | Spatial_KD: {:.4f}".format(
                epoch, iteration, n_batches, batch_loss, loss_triplet.item(), loss_global.item(), loss_spatial.item()), flush=True)
            step = (epoch * n_batches) + iteration
            writer.add_scalar('Train/Loss', batch_loss, step)
            writer.add_scalar('Train/Triplet', loss_triplet.item(), step)
            writer.add_scalar('Train/Global_KD', loss_global.item(), step)
            writer.add_scalar('Train/Spatial_KD', loss_spatial.item(), step)

    avg_loss = epoch_loss / n_batches
    print("===> Epoch {} 蒸餾完成! 平均 Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def infer(eval_set, model_ptr, opt, device):
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False)
    model_ptr.eval()
    num_samples = len(eval_set)
    global_dim = model_ptr.global_feat_dim  
    all_global_descs = np.zeros((num_samples, global_dim), dtype=np.float32)
    
    with torch.no_grad():
        for idx, (imgs, _) in enumerate(tqdm(test_data_loader, desc="Extracting")):
            imgs = imgs.to(device)
            _, _, global_desc = model_ptr(imgs)
            start_idx = idx * opt.cacheBatchSize
            end_idx = start_idx + imgs.shape[0]
            all_global_descs[start_idx:end_idx] = global_desc.detach().cpu().numpy()
                
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_global_descs

if __name__ == "__main__":
    opt = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"===> 載入的 REIN.py: {rein_module.__file__}")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    print('===> 載入預訓練的 ResNet34 + NetVLAD 老師模型')
    teacher = REIN().to(device)
    
    if not isfile(opt.teacher_path):
        kaggle_root = '/kaggle/input'
        found = None
        if exists(kaggle_root):
            for root, dirs, files in os.walk(kaggle_root):
                if 'model_best.pth.tar' in files:
                    found = os.path.join(root, 'model_best.pth.tar')
                    break
        if found:
            print(f"老師模型路徑不存在，已自動找到: {found}")
            opt.teacher_path = found

    if isfile(opt.teacher_path):
        checkpoint = torch.load(opt.teacher_path, map_location=device, weights_only=False)
        teacher.load_state_dict(checkpoint['state_dict'])
        for param in teacher.parameters():
            param.requires_grad = False
        print(f"成功載入老師模型權重: {opt.teacher_path}")
    else:
        raise FileNotFoundError(f"未能在該路徑找到老師模型權重，請確認路徑設定: {opt.teacher_path}")
    teacher.eval()

    print('===> 初始化 MobileNetV3 + Spatial Attention 學生模型')
    student = StudentREIN(teacher_global_dim=8192, feat_h=40, feat_w=40).to(device)
    if not hasattr(student, 'global_pool'):
        raise RuntimeError('目前載入的 StudentREIN 不含 global_pool 修正，請確認 Kaggle 執行的是更新後的 REIN.py')

    writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))
    logdir = writer.file_writer.get_logdir()
    if not exists(logdir): makedirs(logdir)

    print(f"===> 使用 KITTI 路徑: {opt.kitti_path}")
    print(f"===> 使用 NCLT 路徑: {opt.nclt_path}")

    train_set = kitti_dataset.TrainingDataset(dataset_path=opt.kitti_path) 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=opt.lr)    
    criterion = TripletLoss().to(device)
    
    # 🔥 核心修正：將 reduction 改為 'sum'，治本解除維度稀釋問題
    distill_criterion = nn.MSELoss(reduction='sum').to(device) 
    best_score = 0

    print("===> 開始執行知識蒸餾訓練流程")
    for epoch in range(opt.nEpochs):
        train_epoch_distill(epoch, teacher, student, train_set, optimizer, criterion, distill_criterion, writer, device, opt)
        
        # 測試學生的 KITTI 表現
        recalls_kitti = []
        for seq in ['00', '02', '05', '06', '08']:
            test_set = kitti_dataset.InferDataset(seq=seq, dataset_path=opt.kitti_path)   
            global_descs = infer(test_set, student, opt, device)
            recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set)
            recalls_kitti.append(recall_top1)
            writer.add_scalars('val', {'KITTI_' + seq: recall_top1}, epoch)
            del global_descs, test_set; gc.collect()

        # 測試學生的 NCLT 表現
        eval_seq = ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
        eval_datasets = []
        eval_global_descs = []
        for seq in eval_seq:   
            test_set = nclt_dataset.InferDataset(seq=seq, dataset_path=opt.nclt_path)   
            global_descs = infer(test_set, student, opt, device)
            eval_global_descs.append(global_descs)
            eval_datasets.append(test_set)
        recalls_nclt = nclt_dataset.evaluateResults(eval_global_descs, eval_datasets)
        
        for ii in range(len(recalls_nclt)):
            writer.add_scalars('val', {'NCLT_' + eval_seq[ii]: recalls_nclt[ii]}, epoch)
        
        mean_recall = np.mean(recalls_nclt)
        print(f"=== Epoch {epoch} 結束 === 學生 NCLT 平均 Recall@1: {mean_recall*100:.2f}%")
        del eval_global_descs, eval_datasets; gc.collect()

        is_best = mean_recall > best_score 
        if_best: 
            best_score = mean_recall
        
        filename = logdir + '/checkpoint.pth.tar'
        torch.save({
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'recalls': mean_recall,
                'best_score': best_score,
                'optimizer': optimizer.state_dict(),
        }, filename)
        if is_best:
            shutil.copyfile(filename, logdir + '/model_best.pth.tar')

    writer.close()
