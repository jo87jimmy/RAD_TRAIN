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
from model_unet import AnomalyDetectionModel
from data_loader import MVTecDRAEMTrainDataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
def get_available_gpu():
    """自動選擇記憶體使用率最低的GPU"""
    if not torch.cuda.is_available():
        return -1  # 沒有GPU可用

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return -1

    # 檢查每個GPU的記憶體使用情況
    gpu_memory = []
    for i in range(gpu_count):
        torch.cuda.set_device(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_reserved = torch.cuda.memory_reserved(i)
        gpu_memory.append((i, memory_allocated, memory_reserved))

    # 選擇記憶體使用最少的GPU
    available_gpu = min(gpu_memory, key=lambda x: x[1])[0]
    return available_gpu

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def predict_and_visualize_heatmap(model, image_path, device,save_path):
    # ... (前面的預處理部分保持不變) ...
    preprocess = transforms.Compose([...])
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        recon_image_tensor, seg_map_logits = model(image_tensor, return_feats=False)

    # --- 關鍵偵錯步驟 ---
    # 1. 將 logits 轉換為機率
    seg_map_probs = torch.softmax(seg_map_logits, dim=1)

    # 2. 提取 "異常類別" 的機率圖
    # 假設索引 1 代表 "異常" (索引 0 代表 "正常")
    # 如果您的標籤定義相反，請將 [0, 1, :, :] 改為 [0, 0, :, :]
    anomaly_heatmap = seg_map_probs[0, 1, :, :].cpu().numpy()

    # 3. 產生最終的二值化遮罩 (用於比較)
    anomaly_mask = np.argmax(seg_map_probs.cpu().numpy(), axis=1).squeeze()

    # ... (反正規化 recon_image_np 的部分保持不變) ...
    original_image_np = np.array(image.resize((224, 224)))
    # ...

    # --- 可視化 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 顯示原始的異常熱圖
    im = axes[1].imshow(anomaly_heatmap, cmap='viridis') # 使用 viridis 或 jet 色彩映射
    axes[1].set_title('Anomaly Probability Heatmap')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1]) # 加上顏色條以觀察機率範圍

    # 顯示最終的二值化遮罩
    axes[2].imshow(original_image_np)
    axes[2].imshow(anomaly_mask, cmap='jet', alpha=0.5)
    axes[2].set_title('Final ArgMax Mask')
    axes[2].axis('off')
    print(f"Saving out_image to: {save_path}")  # 除錯訊息
    plt.savefig(save_path)

    plt.show()

    # 打印一些數值幫助判斷
    print(f"Heatmap Min Value: {anomaly_heatmap.min()}")
    print(f"Heatmap Max Value: {anomaly_heatmap.max()}")
    print(f"Heatmap Mean Value: {anomaly_heatmap.mean()}")


# =======================
# Main Pipeline
# =======================
def main(obj_names, args):
    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for obj_name in obj_names:
        # Load teacher
        path_run_name = f'./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_'+obj_name+'_'
        checkpoint_path = path_run_name + ".pckl"
        teacher_ckpt = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True)
        # 假設是處理3通道的RGB圖像
        IMG_CHANNELS = 3
        # 分割任務是二分類 (異常 vs. 正常)
        SEG_CLASSES = 2
        # 建立教師模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
        teacher_model = AnomalyDetectionModel(
            recon_in=IMG_CHANNELS,
            recon_out=IMG_CHANNELS,
            recon_base=128,  # 教師重建網路較寬
            disc_in=IMG_CHANNELS * 2, # 原圖+重建圖
            disc_out=SEG_CLASSES,
            disc_base=128    # 教師判別網路較寬
        ).to(device)
        # 檢查 checkpoint 結構
        print("Checkpoint keys:", teacher_ckpt.keys())

        # 將教師模型的參數載入至模型中，使用 checkpoint 中的 'reconstructive' 欄位
        teacher_model.load_state_dict(teacher_ckpt,strict=False)
        # 將教師模型設為評估模式，停用 Dropout、BatchNorm 等訓練專用機制
        teacher_model.eval()

        # 將教師模型的所有參數設為不可訓練，避免在後續訓練中被更新
        for p in teacher_model.parameters():
            p.requires_grad = False

        # Student model
        #dropout 防止過擬合，幫助學生模型泛化，避免過擬合教師模型提取的特徵。在蒸餾訓練時，讓學生模型學到更穩健的特徵，而不是完全模仿教師模型的單一路徑
        student_model = AnomalyDetectionModel(
            recon_in=IMG_CHANNELS,
            recon_out=IMG_CHANNELS,
            recon_base=64,   # 學生重建網路較窄
            disc_in=IMG_CHANNELS * 2, # 原圖+重建圖
            disc_out=SEG_CLASSES,
            disc_base=64     # 學生判別網路較窄
        ).to(device)
        # --- 特徵對齊層 ---
        # 因為判別網路的特徵維度不同，需要對齊層來計算蒸餾損失
        # 學生判別網路的通道數
        s_channels = [64, 128, 256, 512, 512, 512]
        # 教師判別網路的通道數
        t_channels = [128, 256, 512, 1024, 1024, 1024]

        feature_aligns = nn.ModuleList([
            nn.Conv2d(s_c, t_c, kernel_size=1, bias=False)
            for s_c, t_c in zip(s_channels, t_channels)
        ]).to(device)

        student_model.apply(weights_init)

        # 假設 optimizer 只優化 student_model 和 feature_aligns 的參數
        optimizer = torch.optim.Adam(
            list(student_model.parameters()) + list(feature_aligns.parameters()),
            lr=args.lr
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [args.epochs * 0.8, args.epochs * 0.9],
            gamma=0.2,
            last_epoch=-1)
        # 定義損失函數
        # 這裡使用 L2 損失（均方誤差）、SSIM 損失與 Focal Loss
        # loss_l2 = torch.nn.modules.loss.MSELoss()
        # loss_ssim = SSIM()
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

        # 開始進行多輪訓練迴圈
        n_iter = 0

        # --- 超參數定義 ---
        # 設定不同損失的權重

        # 原始分割損失的權重 (監督學生學習真實標籤)
        # 這通常是最主要的目標，可以設為 1.0 作為基準
        lambda_orig_seg = 1.0

        # 分割蒸餾損失的權重 (監督學生模仿教師的分割結果)
        # 這個也很重要，讓學生學習教師的 "軟標籤"，可以設為與原始損失相同或稍低的權重
        lambda_seg_distill = 1.0

        # 特徵蒸餾損失的權重 (監督學生模仿教師的中間特徵)
        # 這是一個輔助和正規化的目標，幫助學生學習更通用的特徵表示。
        # 由於多層特徵的損失加總後數值可能較大，通常會給予一個較小的權重。
        lambda_feat_distill = 0.5
        # 假設 teacher_model 和 student_model 已經被定義和加載
        # teacher_model 包含重建和判別子網路 (TRecon, TDisc)
        # student_model 包含重建和判別子網路 (SRecon, SDisc)
        # 假設 lambda_feat_distill, lambda_seg_distill, lambda_orig_seg 已經定義
        for epoch in range(args.epochs):
            print("Epoch: " + str(epoch))
            for i_batch, sample_batched in enumerate(train_loader):
                # 遍歷訓練資料集的每個批次
                input_image = sample_batched["image"].to(device)
                # 真實的異常遮罩，用於計算原始分割損失
                ground_truth_mask = sample_batched["anomaly_mask"].to(device)
                aug_gray_batch = sample_batched["augmented_image"].to(device)

                # --- 教師網路前向傳播 (不計算梯度) ---
                with torch.no_grad():
                   _, teacher_seg_map, teacher_features = teacher_model(aug_gray_batch, return_feats=True)

                # --- 學生網路前向傳播 ---
                _, student_seg_map, student_features = student_model(aug_gray_batch, return_feats=True)

                # --- 計算損失函數 ---

                # 1. 特徵蒸餾損失
                feat_distill_loss = 0.0
                for i in range(len(student_features)):
                    # 將學生特徵對齊到教師特徵的維度
                    aligned_student_feat = feature_aligns[i](student_features[i])
                    feat_distill_loss += F.mse_loss(
                        F.normalize(aligned_student_feat, p=2, dim=1),
                        F.normalize(teacher_features[i], p=2, dim=1)
                    )

                # 2. 分割蒸餾損失 (Segmentation Distillation Loss)
                # 讓學生的分割圖模仿教師的分割圖
                # 這裡以 KL 散度為例，教師的輸出需要先經過 softmax 轉換成機率分佈
                teacher_seg_log_softmax = F.log_softmax(teacher_seg_map, dim=1)
                student_seg_log_softmax = F.log_softmax(student_seg_map, dim=1)
                # KL 散度損失
                seg_distill_loss = F.kl_div(student_seg_log_softmax, teacher_seg_log_softmax.exp(), reduction='batchmean')
                # 或者，您也可以使用 MSE 損失:
                # seg_distill_loss = F.mse_loss(student_seg_map, teacher_seg_map)

                # 3. 原始分割損失 (Original Segmentation Loss)
                # 使用真實的異常遮罩監督學生的分割結果
                # 以焦點損失 (Focal Loss) 為例
                student_seg_softmax = torch.softmax(student_seg_map, dim=1)
                orig_seg_loss = loss_focal(student_seg_softmax, ground_truth_mask)


                # --- 判別網路總損失 ---
                # --- 總損失與更新 ---
                total_loss = (lambda_feat_distill * feat_distill_loss +
                            lambda_seg_distill * seg_distill_loss +
                            lambda_orig_seg * orig_seg_loss)

                # --- 反向傳播與參數更新 ---
                # 清除先前計算的梯度
                optimizer.zero_grad()
                # 計算梯度
                total_loss.backward()
                # 更新學生判別網路 (以及重建網路) 的權重
                optimizer.step()

                # 記錄訓練過程
                writer.add_scalar("Train/Total_Loss", total_loss.item(), n_iter)
                writer.add_scalar("Train/Feature_Distillation_Loss", feat_distill_loss.item(), n_iter)
                writer.add_scalar("Train/Segmentation_Distillation_Loss", seg_distill_loss.item(), n_iter)
                writer.add_scalar("Train/Original_Segmentation_Loss", orig_seg_loss.item(), n_iter)
                predict_and_visualize_heatmap(student_model, sample_batched["image"], device,save_root)
                n_iter += 1

            # 每個 epoch 結束後更新學習率並保存模型
            scheduler.step()
            torch.save(student_model.state_dict(), os.path.join(checkpoint_dir, obj_name+".pckl"))

        # 關閉 TensorBoard 紀錄器，釋放資源
        writer.close()
        torch.cuda.empty_cache()

# =======================
# Run pipeline
# =======================
if __name__ == "__main__":
    """
    --gpu_id -2：自動選擇最佳GPU
    --gpu_id -1：強制使用CPU
    --gpu_id  0：使用GPU 0（原有行為）
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=-2, required=False,
                    help='GPU ID (-2: auto-select, -1: CPU)')
    args = parser.parse_args()

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [
        ['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
        ['transistor'], ['tile'], ['cable'], ['zipper'], ['toothbrush'],
        ['metal_nut'], ['hazelnut'], ['screw'], ['grid'], ['wood']
    ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule', 'bottle', 'carpet', 'leather', 'pill',
                    'transistor', 'tile', 'cable', 'zipper', 'toothbrush',
                    'metal_nut', 'hazelnut', 'screw', 'grid', 'wood']
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    # 根據選擇的GPU執行
    if args.gpu_id == -1:
        # 使用CPU
        main(picked_classes, args)
    else:
        # 使用GPU
        with torch.cuda.device(args.gpu_id):
            main(picked_classes, args)
