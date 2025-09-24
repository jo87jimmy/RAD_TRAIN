import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import random  # 亂數控制
import argparse  # 命令列參數處理
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from loss import FocalLoss, SSIM
from model_unet import ReconstructiveSubNetwork, StudentReconstructiveSubNetwork, DiscriminativeSubNetwork
from data_loader import MVTecDRAEMTrainDataset


def setup_seed(seed):
    # 設定隨機種子，確保實驗可重現
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保證結果可重現
    torch.backends.cudnn.benchmark = False  # 關閉自動最佳化搜尋

# =======================
# Utilities
# =======================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# =======================
# Main Pipeline
# =======================
def main(obj_names, args):
    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for obj_name in obj_names:
        # Load teacher
        # 載入教師模型的檢查點（checkpoint）檔案，並指定載入到的裝置（如 GPU 或 CPU）
        path_run_name = f'./DRAEM_seg_large_ae_large_0.0001_800_bs8_'+obj_name+'_'
        teacher_ckpt = torch.load(
            path_run_name+".pckl",
            map_location=device,
            weights_only=True)
        # 建立教師模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
        teacher_model = ReconstructiveSubNetwork(in_channels=3,
                                                out_channels=3).to(device)
        # 將教師模型的參數載入至模型中，使用 checkpoint 中的 'reconstructive' 欄位
        teacher_model.load_state_dict(teacher_ckpt)
        # 將教師模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
        teacher_model.eval()
        # 將教師模型的所有參數設為不可訓練，避免在後續訓練中被更新
        for p in teacher_model.parameters():
            p.requires_grad = False

        # Student model
        #dropout 防止過擬合，幫助學生模型泛化，避免過擬合教師模型提取的特徵。在蒸餾訓練時，讓學生模型學到更穩健的特徵，而不是完全模仿教師模型的單一路徑
        student_model = StudentReconstructiveSubNetwork(
            in_channels=3,
            out_channels=3,
            base_width=64,  # 壓縮後的維度
            teacher_base_width=128  # 教師模型的維度
        ).to(device)
        student_model.apply(weights_init)
        student_seg = DiscriminativeSubNetwork(in_channels=6,
                                            out_channels=2).to(device)
        student_seg.apply(weights_init)
        #定義學生模型優化器和學習率排程器
        # optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

        optimizer = torch.optim.Adam([{
            "params": student_model.parameters(),
            "lr": args.lr
        }, {
            "params": student_seg.parameters(),
            "lr": args.lr
        }])

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [args.epochs * 0.8, args.epochs * 0.9],
            gamma=0.2,
            last_epoch=-1)
        # 定義損失函數
        # 這裡使用 L2 損失（均方誤差）、SSIM 損失與 Focal Loss
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        path = f'./mvtec'  # 訓練資料路徑
        path_dtd = f'./dtd/images/'
        # Load datasets
        # 載入訓練資料集，指定根目錄、類別、資料切分方式為 "train"，並將影像尺寸調整為 256x256
        #python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/

        train_dataset = MVTecDRAEMTrainDataset(root_dir=path + f'/{obj_name}/train/good/',anomaly_source_path=path_dtd,
                                    resize_shape=[256, 256])
        # 建立訓練資料的 DataLoader，設定每批次大小為 16，打亂資料順序，使用 4 個執行緒加速載入
        train_loader = DataLoader(train_dataset,
                                batch_size=args.bs,
                                shuffle=True,
                                num_workers=4)
        # 主儲存資料夾路徑
        save_root = "./save_files"

        # 若主資料夾不存在，則建立
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        # TensorBoard
        # 建立 TensorBoard 的紀錄器，將訓練過程的指標與圖像輸出到指定目錄 "./save_files"
        writer = SummaryWriter(log_dir=save_root)
        # 指定模型檢查點（checkpoint）儲存的資料夾路徑
        # 模型檢查點儲存路徑
        checkpoint_dir = os.path.join(save_root, "checkpoints")
        # 如果檢查點資料夾不存在，則建立該資料夾（exist_ok=True 表示若已存在則不報錯）
        os.makedirs(checkpoint_dir, exist_ok=True)

        lambda_distill = 1.0  # 蒸餾 loss 權重，可依需求調整

        # 開始進行多輪訓練迴圈
        n_iter = 0
        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(train_loader):
                # 遍歷訓練資料集的每個批次
                gray_batch = sample_batched["image"].to(device)
                aug_gray_batch = sample_batched["augmented_image"].to(device)
                anomaly_mask = sample_batched["anomaly_mask"].to(device)

                # 使用教師模型提取特徵，並停用梯度計算以節省記憶體與加速推論
                # --- 教師模型 forward（不計梯度） ---
                with torch.no_grad():
                    _, teacher_feats = teacher_model(gray_batch, return_feats=True)

                # --- 學生模型 forward ---
                recon_output, student_feats = student_model(gray_batch)
                # Hard loss (重建誤差)
                recon_loss = F.mse_loss(recon_output, gray_batch)

                # Soft loss: 初始化 distill_loss
                distill_loss = 0.0
                # Soft loss 學生同時學到 低階特徵 (邊緣/紋理) 和 高階語義特徵
                for t_feat, s_feat in zip(teacher_feats, student_feats):
                    # normalize 避免只學到幅度
                    distill_loss += F.mse_loss(F.normalize(s_feat, dim=1),
                                            F.normalize(t_feat, dim=1))

                # --- 總 loss ---
                loss = recon_loss + lambda_distill * distill_loss

                joined_in = torch.cat((recon_output, aug_gray_batch), dim=1)

                out_mask = student_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                # 重建圖與原始圖差異
                l2_loss = loss_l2(recon_output,gray_batch)
                # 計算重建圖與原始圖的 SSIM
                ssim_loss = loss_ssim(recon_output, gray_batch)

                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss


                # 清除先前的梯度
                optimizer.zero_grad()
                # 反向傳播計算梯度
                loss.backward()
                # 更新學生模型的參數
                optimizer.step()
                # 將訓練損失記錄到 TensorBoard，標記為 "Train/Loss"
                writer.add_scalar("Train/Loss", loss.item(), n_iter)
                # 步數累加，用於追蹤訓練進度
                n_iter +=1

            scheduler.step()
            torch.save(student_model.state_dict(), os.path.join(checkpoint_dir, obj_name+".pckl"))
            torch.save(student_seg.state_dict(), os.path.join(checkpoint_dir, obj_name+"_seg.pckl"))

        # 關閉 TensorBoard 紀錄器，釋放資源
        writer.close()
        torch.cuda.empty_cache()

# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    # 解析命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)# 訓練類別
    parser.add_argument('--epochs', default=25, type=int)  # 訓練回合數
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    args = parser.parse_args()

    obj_batch = [['capsule'],
                ['bottle'],
                ['carpet'],
                ['leather'],
                ['pill'],
                ['transistor'],
                ['tile'],
                ['cable'],
                ['zipper'],
                ['toothbrush'],
                ['metal_nut'],
                ['hazelnut'],
                ['screw'],
                ['grid'],
                ['wood']
                ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        main(picked_classes, args)
