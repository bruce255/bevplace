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
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

from sklearn.decomposition import PCA

from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm
import faiss
from uuid import uuid4

import kitti_dataset
import nclt_dataset 

def get_args():
    parser = argparse.ArgumentParser(description='BEVPlace++')
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test'])
    parser.add_argument('--batchSize', type=int, default=4, help='Number of triplets')
    parser.add_argument('--cacheBatchSize', type=int, default=128, help='Batch size for caching and testing')
    parser.add_argument('--nEpochs', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    parser.add_argument('--lrStep', type=float, default=10, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--threads', type=int, default=24, help='Number of threads for data loader')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')
    parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
    parser.add_argument('--cachePath', type=str, default='./cache/', help='Path to save cache to.')
    parser.add_argument('--load_from', type=str, default='', help='Path to load checkpoint.')
    parser.add_argument('--ckpt', type=str, default='best', choices=['latest', 'best'])
    
    opt = parser.parse_args()
    return opt
#=====================================================================================================================================#
#                                                               三源組損失函數                                                          #
#=====================================================================================================================================#

# 說明：Triplet Loss 是一種常用於度量學習的損失函數，旨在將相似的樣本拉近，將不相似的樣本推遠。它通常使用三個樣本：anchor（錨點）、positive（正樣本）和negative（負樣本）。目標是使 anchor 與 positive 的距離小於 anchor 與 negative 的距離，並且至少有一個 margin（邊界）。

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss
#=====================================================================================================================================#
# 在 CARLA 裡，train_epoch 就像一次完整的「自駕車訓練遊戲回合」，從場景資料、模型推理、評分，到優化更新，最後把結果記錄下來。
#=====================================================================================================================================#
def train_epoch(epoch, model, train_set, optimizer, criterion, writer, device, opt):
    #第0回合用於初始化
    epoch_loss = 0
    #計算幾個batchh需要訓練完一輪，這裡是根據訓練集的大小和每批次的大小來計算的，確保即使最後一批次不滿，也能正確處理。
    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize
    
    # 1. Building Cache (優化：顯存清理)
    if epoch >= 0:
        print('====> Building Cache for Hard Mining')
        #因為預設的 train_set.mining 是 True，這裡先設為 False，避免在建立緩存時進行困難樣本挖掘。
        train_set.mining = False
        #建立cache的路徑，這裡是將 opt.cachePath 與 'train_feat_cache.hdf5' 組合成完整的路徑，用於存儲訓練特徵的緩存文件。
        train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
        
        if not exists(opt.cachePath):
            makedirs(opt.cachePath)
        #建立新的 HDF5 檔案，並在其中創建一個名為 "features" 的資料集，用於存儲訓練樣本的全局特徵向量。 
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = model.global_feat_dim
            #在剛開的 HDF5 檔案裡建立一個名叫 features 的資料集（類似 Excel 裡的一個工作表或表格）
            h5feat = h5.create_dataset("features", [len(train_set), pool_size], dtype=np.float32)
            #DataLoader = 自動幫你分批、打亂、多執行緒預載的資料管線  自動打包要處理的資料=>加速用
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                             batch_size=opt.cacheBatchSize, shuffle=False, 
                                             collate_fn=kitti_dataset.collate_fn)
            model.eval()
            #這個程式碼段落是在建立特徵緩存，為後續的困難樣本挖掘（Hard Mining）做準備。
            with torch.no_grad():
                for iteration, (query, _, _, indices) in enumerate(training_data_loader, 1):
                    query = query.to(device)
                    _, _, global_descs = model(query)
                    #將 global_descs 的 NumPy 版本，寫入 h5feat 的 indices 這些列，覆蓋對應的所有欄位。
                    h5feat[indices, :] = global_descs.detach().cpu().numpy()
                    
        train_set.mining = True
        train_set.refreshCache()
        torch.cuda.empty_cache()

    # 2. 正式訓練循環 (優化：分批前向傳播，避免 cat 導致 OOM)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                     batch_size=opt.batchSize, shuffle=True, 
                                     collate_fn=kitti_dataset.collate_fn)
    model.train()

    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader):
        B = query.shape[0]
        num_negs = negatives.shape[0] // B

        # 這裡改成單獨 forward 提取，或者分開送入，避免一次過大
        query = query.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        optimizer.zero_grad()

        # 分別 forward，大幅度節省疊加帶來的顯存峰值
        _, _, global_descs_Q = model(query)
        _, _, global_descs_P = model(positives)
        _, _, global_descs_N = model(negatives)

        loss = 0
        for i in range(B):
            max_loss = torch.max(criterion(
                global_descs_Q[i], 
                global_descs_P[i], 
                global_descs_N[num_negs * i : num_negs * (i + 1)]
            ))
            loss += max_loss
        
        loss /= B
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, n_batches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * n_batches) + iteration)

    avg_loss = epoch_loss / n_batches
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
    
    # 訓練完一輪立刻釋放顯存
    torch.cuda.empty_cache()

def infer(eval_set, model_ptr, opt, device, return_local_feats=False):
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads,
                                  batch_size=opt.cacheBatchSize, shuffle=False)
    model_ptr.eval()

    num_samples = len(eval_set)
    global_dim = 512 * 128

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    # create memmap files to avoid allocating large arrays in RAM
    gpath = join(opt.cachePath, f"global_descs_{uuid4().hex}.npy")
    all_global_descs = np.memmap(gpath, dtype='float32', mode='w+', shape=(num_samples, global_dim))

    all_local_feats = None
    lpath = None

    with torch.no_grad():
        for idx, (imgs, _) in enumerate(tqdm(test_data_loader, desc="Extracting")):
            imgs = imgs.to(device)
            _, local_feat, global_desc = model_ptr(imgs)

            start_idx = idx * opt.cacheBatchSize
            end_idx = start_idx + imgs.shape[0]

            all_global_descs[start_idx:end_idx] = global_desc.detach().cpu().numpy()

            if return_local_feats:
                local_feat_np = local_feat.detach().cpu().numpy()
                if all_local_feats is None:
                    local_shape = (num_samples, *local_feat_np.shape[1:])
                    lpath = join(opt.cachePath, f"local_feats_{uuid4().hex}.npy")
                    all_local_feats = np.memmap(lpath, dtype='float32', mode='w+', shape=local_shape)
                all_local_feats[start_idx:end_idx] = local_feat_np

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if return_local_feats:
        return all_local_feats, all_global_descs
    else:
        return all_global_descs

def getClusters(cluster_set, model, device, opt):
    n_descriptors = 10000
    n_per_image = 25
    n_im = ceil(n_descriptors / n_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    data_loader = DataLoader(dataset=cluster_set, num_workers=opt.threads, 
                             batch_size=opt.cacheBatchSize, shuffle=False, sampler=sampler)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    initcache = join(opt.cachePath, 'desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            all_feats = h5.create_dataset("descriptors", [n_descriptors, 128], dtype=np.float32)

            for iteration, (query, _, _, _) in enumerate(data_loader, 1):
                query = query.to(device)
                local_feat, _, _ = model(query)
                local_feat = local_feat.view(query.size(0), 128, -1).permute(0, 2, 1)
                
                batchix = (iteration - 1) * opt.cacheBatchSize * n_per_image
                for ix in range(local_feat.size(0)):
                    sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
                    startix = batchix + ix * n_per_image
                    all_feats[startix:startix + n_per_image, :] = local_feat[ix, sample, :].detach().cpu().numpy()

        print('====> Clustering..')
        # Sample a subset of descriptors for kmeans training to avoid loading everything
        sample_n = min(5000, n_descriptors)
        sample_idx = np.random.choice(n_descriptors, sample_n, replace=False)
        sample_feats = all_feats[sample_idx]

        kmeans = faiss.Kmeans(128, 64, niter=100, verbose=False)
        kmeans.train(sample_feats)
        h5.create_dataset('centroids', data=kmeans.centroids)

def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    filename = model_out_path + '/' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_out_path + '/' + 'model_best.pth.tar')

if __name__ == "__main__":
    opt = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    print('===> Building model')
    from REIN import REIN
    model = REIN().to(device)
    
    if opt.load_from != '':
        resume_ckpt = join(opt.load_from, 'model_best.pth.tar' if opt.ckpt.lower() == 'best' else 'checkpoint.pth.tar')
        if isfile(resume_ckpt):
            print(f"=> loading checkpoint '{resume_ckpt}'")
            checkpoint = torch.load(resume_ckpt, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"=> no checkpoint found at '{resume_ckpt}'")
    else:
        initcache = join(opt.cachePath, 'desc_cen.hdf5')
        if not isfile(initcache):
            train_set = kitti_dataset.TrainingDataset()
            getClusters(train_set, model, device, opt)
        with h5py.File(initcache, mode='r') as h5: 
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pooling.init_params(clsts, traindescs) 
            model = model.to(device)

    if opt.mode.lower() == 'train':
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))
        logdir = writer.file_writer.get_logdir()
        if not exists(logdir): makedirs(logdir)

        with open(join(logdir, 'flags.json'), 'w') as f:
            f.write(json.dumps({k: v for k, v in vars(opt).items()}))

        train_set = kitti_dataset.TrainingDataset() 
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)    
        criterion = TripletLoss().to(device)
        best_score = 0

        for epoch in range(opt.nEpochs):
            train_epoch(epoch, model, train_set, optimizer, criterion, writer, device, opt)
            
            # 訓練中的簡化測試（避免多序列堆疊 OOM）
            recalls_kitti = []
            for seq in ['00', '02', '05', '06', '08']:
                test_set = kitti_dataset.InferDataset(seq=seq)   
                global_descs = infer(test_set, model, opt, device)
                recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set)
                recalls_kitti.append(recall_top1)
                writer.add_scalars('val', {'KITTI_' + seq: recall_top1}, epoch)
                del global_descs, test_set; gc.collect()

            # NCLT 訓練中測試優化：不採用 append 堆疊，直接一條一條處理並拿 recall
            eval_seq = ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
            # Use first sequence as the database and process queries one-by-one
            db_seq = eval_seq[0]
            db_set = nclt_dataset.InferDataset(seq=db_seq)
            db_descs = infer(db_set, model, opt, device)
            recalls_nclt = []
            for seq in eval_seq[1:]:
                test_set = nclt_dataset.InferDataset(seq=seq)
                q_descs = infer(test_set, model, opt, device)
                r = nclt_dataset.evaluateResults([db_descs, q_descs], [db_set, test_set])
                recalls_nclt.append(r[0] if len(r) > 0 else 0.0)
                writer.add_scalars('val', {'NCLT_' + seq: recalls_nclt[-1]}, epoch)
                del q_descs, test_set; gc.collect()

            mean_recall = np.mean(recalls_nclt)
            del db_descs, db_set; gc.collect()

            is_best = mean_recall > best_score 
            if is_best: best_score = mean_recall
            
            saveCheckpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': mean_recall,
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
            }, is_best, logdir)

        writer.close()

    elif opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        recalls_kitti = []
        eval_seq_kitti = ['08']

        for seq in eval_seq_kitti:   
            if seq == '08':
                test_set = kitti_dataset.InferDataset(seq=seq, sample_inteval=5)  
                local_feats, global_descs = infer(test_set, model, opt, device, return_local_feats=True)  
                recall_top1, success_rate, mean_trans_err, mean_rot_err = kitti_dataset.evaluateResults(seq, global_descs, local_feats, test_set, "out_imgs/")
                del local_feats
            else:
                test_set = kitti_dataset.InferDataset(seq=seq)  
                global_descs = infer(test_set, model, opt, device)
                recall_top1 = kitti_dataset.evaluateResults(seq, global_descs, None, test_set)
            recalls_kitti.append(recall_top1)
            del global_descs, test_set; gc.collect()

        print('====> Extracting Features of NCLT (優化版防爆)')
        eval_seq = ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
        # Build index from the first sequence and search each remaining sequence sequentially
        db_seq = eval_seq[0]
        print(f'Building NCLT index from {db_seq}')
        db_set = nclt_dataset.InferDataset(seq=db_seq)
        db_descs = infer(db_set, model, opt, device)
        recalls_nclt = []
        for seq in eval_seq[1:]:
            print(f'Processing NCLT sequence: {seq}')
            test_set = nclt_dataset.InferDataset(seq=seq)
            q_descs = infer(test_set, model, opt, device)
            r = nclt_dataset.evaluateResults([db_descs, q_descs], [db_set, test_set])
            recalls_nclt.append(r[0] if len(r) > 0 else 0.0)
            del q_descs, test_set; torch.cuda.empty_cache(); gc.collect()
        print('NCLT Mean Recall: %0.2f' % (np.mean(recalls_nclt) * 100))
        del db_descs, db_set; gc.collect()