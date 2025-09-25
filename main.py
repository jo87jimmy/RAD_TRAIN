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
from datetime import datetime


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
    """ 卷積層權重 → 小亂數讓網路容易學習、梯度穩定
        BatchNorm權重 → 初始縮放 1 保持數值穩定，偏置 0 不改變均值"""
    # 取得模組的類別名稱，例如 'Conv2d', 'BatchNorm2d' 等
    classname = m.__class__.__name__

    # 如果是卷積層 (Conv)
    if classname.find('Conv') != -1:
        # 權重初始化為均值 0、標準差 0.02 的高斯分布
        # 原因：這樣可以讓卷積層一開始輸出的特徵值分布均衡，避免梯度消失或梯度爆炸
        m.weight.data.normal_(0.0, 0.02)

    # 如果是批次正規化層 (BatchNorm)
    elif classname.find('BatchNorm') != -1:
        # 權重初始化為均值 1、標準差 0.02 的高斯分布
        # 原因：BatchNorm 的權重 (gamma) 控制輸出縮放，初始化為 1 可以保持初始特徵分布不變
        m.weight.data.normal_(1.0, 0.02)
        # 偏置初始化為 0
        # 原因：偏置 (beta) 控制輸出偏移量，初始化為 0 可以保持輸出均值不偏移
        m.bias.data.fill_(0)


def predict_and_visualize_heatmap(model, image_input, device, save_path):
    """
    修改後的函數，能夠同時處理文件路徑和圖像 Tensor

    Args:
        model: 訓練好的模型
        image_input: 可以是文件路徑字符串或圖像張量 [batch_size, channels, height, width]
        device: 設備
        save_path: 保存路徑
    """
    # 定義預處理轉換
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 判斷輸入類型並進行相應處理
    if isinstance(image_input, str):
        # 輸入是文件路徑
        image = Image.open(image_input).convert("RGB")
        original_image_np = np.array(image.resize((224, 224)))
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        batch_size = 1
    else:
        # 輸入是 Tensor
        image_tensor = image_input.to(device)
        batch_size = image_tensor.size(0)

        # 將 Tensor 轉換回 PIL 圖像用於可視化
        # 注意：這裡假設 Tensor 是 [C, H, W] 或 [B, C, H, W] 格式，且值在 [0,1] 或已歸一化
        if batch_size == 1:
            img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()
        else:
            img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()

        # 反正規化並轉換到 [0,255] 範圍
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        original_image_np = img_np.astype('uint8')

    model.eval()
    with torch.no_grad():
        recon_image_tensor, seg_map_logits = model(image_tensor,
                                                   return_feats=False)

    # --- 關鍵偵錯步驟 ---
    # 1. 將 logits 轉換為機率
    seg_map_probs = torch.softmax(seg_map_logits, dim=1)

    # 2. 提取 "異常類別" 的機率圖
    # 假設索引 1 代表 "異常" (索引 0 代表 "正常")
    anomaly_heatmap = seg_map_probs[0, 1, :, :].cpu().numpy()

    # 3. 產生最終的二值化遮罩 (用於比較)
    masks = np.argmax(seg_map_probs.cpu().numpy(), axis=1)  # (B, H, W)
    anomaly_mask = masks[0]  # 取第一張 (H, W)
    # anomaly_mask = np.argmax(seg_map_probs.cpu().numpy(), axis=1).squeeze()

    # --- 可視化 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 顯示原始圖像
    axes[0].imshow(original_image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 顯示異常熱圖
    im = axes[1].imshow(anomaly_heatmap, cmap='viridis')
    axes[1].set_title('Anomaly Probability Heatmap')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1])

    # 顯示最終的二值化遮罩
    axes[2].imshow(original_image_np)
    axes[2].imshow(anomaly_mask, cmap='jet', alpha=0.5)
    axes[2].set_title('Final ArgMax Mask')
    axes[2].axis('off')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path_cv = f"{save_path}/out_mask_cv_{timestamp}.png"
    print(f"Saving out_image to: {save_path_cv}")
    plt.savefig(save_path_cv)
    plt.close(fig)  # 關閉圖形以避免記憶體洩漏
    model.train()  # 恢復訓練模式


# =======================
# Main Pipeline
# =======================
def main(obj_names, args):
    setup_seed(111)  # 固定隨機種子
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for obj_name in obj_names:
        # Load teacher
        recon_path = f'./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_' + obj_name + '_'
        checkpoint_path = recon_path + ".pckl"
        teacher_recon_ckpt = torch.load(checkpoint_path,
                                        map_location=device,
                                        weights_only=True)
        print("teacher_recon_ckpt keys:", teacher_recon_ckpt.keys())

        seg_path = f'./DRAEM_checkpoints/DRAEM_seg_large_ae_large_0.0001_800_bs8_' + obj_name + '__seg'
        checkpoint_seg_path = seg_path + ".pckl"
        teacher_seg_ckpt = torch.load(checkpoint_seg_path,
                                      map_location=device,
                                      weights_only=True)
        print("teacher_seg_ckpt keys:", teacher_seg_ckpt.keys())
        # # 合併兩個 state_dict
        # full_ckpt = {}
        # full_ckpt.update(teacher_recon_ckpt)  # encoder/decoder 權重
        # full_ckpt.update(teacher_seg_ckpt)  # discriminator 權重

        # 創建一個新的 state_dict 來存放修正後的鍵
        new_teacher_state_dict = {}

        # 為重建子網路的權重加上 "reconstruction_subnet." 前綴
        for key, value in teacher_recon_ckpt.items():
            new_key = "reconstruction_subnet." + key
            new_teacher_state_dict[new_key] = value

        # 為判別子網路的權重加上 "discriminator_subnet." 前綴
        for key, value in teacher_seg_ckpt.items():
            # 原始的DRAEM checkpoint可能包含 "module." 前綴（如果使用了DataParallel）
            if key.startswith('module.'):
                key = key[7:]
            new_key = "discriminator_subnet." + key
            new_teacher_state_dict[new_key] = value

        # 假設是處理3通道的RGB圖像
        IMG_CHANNELS = 3
        # 分割任務是二分類 (異常 vs. 正常)
        SEG_CLASSES = 2
        # 建立教師模型的結構，輸入與輸出通道皆為 3（RGB），並移動到指定裝置上
        teacher_model = AnomalyDetectionModel(
            recon_in=IMG_CHANNELS,
            recon_out=IMG_CHANNELS,
            recon_base=128,  # 教師重建網路較寬
            disc_in=IMG_CHANNELS * 2,  # 原圖+重建圖
            disc_out=SEG_CLASSES,
            disc_base=64  # 教師判別網路較寬
        ).to(device)
        # # 檢查 checkpoint 結構
        # print("Checkpoint keys:", full_ckpt.keys())

        print("Checkpoint keys:", new_teacher_state_dict.keys())
        # 現在使用修正後的 state_dict 載入，並使用 strict=True 來確保所有權重都正確載入
        teacher_model.load_state_dict(new_teacher_state_dict, strict=True)

        # # 將教師模型的參數載入至模型中，使用 checkpoint 中的 'reconstructive' 欄位
        # teacher_model.load_state_dict(full_ckpt, strict=False)
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
            recon_base=64,  # 學生重建網路較窄
            disc_in=IMG_CHANNELS * 2,  # 原圖+重建圖
            disc_out=SEG_CLASSES,
            disc_base=64  # 學生判別網路較窄
        ).to(device)

        #初始化 卷積層和 BatchNorm 層的初始權重分布合理，幫助模型更快收斂
        student_model.apply(weights_init)

        # --- 特徵對齊層 ---
        # 判別網路的特徵維度不同，需要對齊層來計算蒸餾損失
        # 學生判別網路的通道數
        s_channels = [64, 128, 256, 512, 512, 512]
        # 教師判別網路的通道數
        t_channels = [64, 128, 256, 512, 512, 512]
        # 使用 ModuleList 建立多個 1x1 Conv2d 層，用來將學生特徵對齊到教師特徵
        feature_aligns = nn.ModuleList([
            nn.Conv2d(s_c, t_c, kernel_size=1, bias=False)
            for s_c, t_c in zip(s_channels, t_channels)
        ]).to(device)

        # 定義優化器，只優化學生模型和特徵對齊層的參數
        optimizer = torch.optim.Adam(list(student_model.parameters()) +
                                     list(feature_aligns.parameters()),
                                     lr=args.lr)
        # 設定學習率調整策略，使用 MultiStepLR(一開始大步走，後面小步走)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,  # 需要調整的優化器
            [args.epochs * 0.8, args.epochs * 0.9],  # 在訓練 80% 和 90% 時調整學習率
            gamma=0.2,  # 每次調整時學習率乘上 0.2
            last_epoch=-1)  # 從頭開始計算學習率
        # 定義損失函數
        loss_focal = FocalLoss()  #解決類別不平衡、強化模型對難分類樣本的學習。

        path = f'./mvtec'  # 訓練資料路徑
        path_dtd = f'./dtd/images/'
        # Load datasets
        # 載入訓練資料集，指定根目錄、類別、資料切分方式為 "train"，並將影像尺寸調整為 256x256
        #python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/

        train_dataset = MVTecDRAEMTrainDataset(root_dir=path +
                                               f'/{obj_name}/train/good/',
                                               anomaly_source_path=path_dtd,
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
        lambda_orig_seg = 1.0  # 這通常是最主要的目標，可以設為 1.0 作為基準

        # 分割蒸餾損失的權重 (監督學生模仿教師的分割結果)
        lambda_seg_distill = 1.0  # 讓學生學習教師的 "軟標籤soft target"，可以設為與原始損失相同或稍低的權重

        # 特徵蒸餾損失的權重 (監督學生模仿教師的中間特徵)
        # 這是一個輔助和正規化的目標，幫助學生學習更通用的特徵表示。
        # 由於多層特徵的損失加總後數值可能較大，通常會給予一個較小的權重。
        lambda_feat_distill = 0.5

        for epoch in range(args.epochs):
            print("Epoch: " + str(epoch))
            for i_batch, sample_batched in enumerate(train_loader):
                # 遍歷訓練資料集的每個批次
                input_image = sample_batched["image"].to(device)
                # 真實的異常遮罩，用於計算原始分割損失
                ground_truth_mask = sample_batched["anomaly_mask"].to(device)
                aug_gray_batch = sample_batched["augmented_image"].to(
                    device)  # 增強的灰階圖

                # --- 教師網路前向傳播 (不計算梯度) ---
                with torch.no_grad():
                    _, teacher_seg_map, teacher_features = teacher_model(
                        input_image, return_feats=True)

                # --- 學生網路前向傳播 ---
                student_recon_image, student_seg_map, student_features = student_model(
                    input_image, return_feats=True)

                # --- 計算損失函數 ---

                # 1. 特徵蒸餾損失
                feat_distill_loss = 0.0
                for i in range(len(student_features)):
                    # 將學生特徵對齊到教師特徵的維度
                    aligned_student_feat = feature_aligns[i](
                        student_features[i])
                    feat_distill_loss += F.mse_loss(
                        F.normalize(aligned_student_feat, p=2, dim=1),
                        F.normalize(teacher_features[i], p=2, dim=1))

                # 2. 分割蒸餾損失 (Segmentation Distillation Loss)
                # 讓學生的分割圖模仿教師的分割圖
                # 這裡以 KL 散度為例，教師的輸出需要先經過 softmax 轉換成機率分佈
                teacher_seg_log_softmax = F.log_softmax(teacher_seg_map, dim=1)
                student_seg_log_softmax = F.log_softmax(student_seg_map, dim=1)
                # KL 散度損失
                seg_distill_loss = F.kl_div(student_seg_log_softmax,
                                            teacher_seg_log_softmax.exp(),
                                            reduction='batchmean')
                # 或者，您也可以使用 MSE 損失:
                # seg_distill_loss = F.mse_loss(student_seg_map, teacher_seg_map)

                # 3. 原始分割損失 (Original Segmentation Loss)
                # 使用真實的異常遮罩監督學生的分割結果
                # 以焦點損失 (Focal Loss) 為例
                student_seg_softmax = torch.softmax(student_seg_map, dim=1)
                orig_seg_loss = loss_focal(student_seg_softmax,
                                           ground_truth_mask)

                # 4. 新增：重建損失 (Reconstruction Loss)
                # 這個損失只在輸入是 "正常" 圖像時計算才有意義，
                # 但在 DRAEM 的設定中，我們用 aug_gray_batch，它是有異常的。
                # 正確的做法是讓重建網路去重建原始的、無異常的圖像 input_image

                # 讓學生模型也對原始正常圖像進行重建
                student_recon_normal, _ = student_model(input_image,
                                                        return_feats=False)

                # 計算重建損失，通常使用 L1 或 L2 Loss
                recon_loss = F.l1_loss(student_recon_normal, input_image)
                # 或者 F.mse_loss

                # --- 超參數定義 ---
                lambda_recon = 1.0  # 重建損失的權重，需要調整

                # --- 重建網路的學生判別網路總損失 ---
                # --- 總損失與更新 ---
                total_loss = (lambda_recon * recon_loss +
                              lambda_feat_distill * feat_distill_loss +
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
                writer.add_scalar("Train/Total_Loss", total_loss.item(),
                                  n_iter)
                writer.add_scalar("Train/Feature_Distillation_Loss",
                                  feat_distill_loss.item(), n_iter)
                writer.add_scalar("Train/Segmentation_Distillation_Loss",
                                  seg_distill_loss.item(), n_iter)
                writer.add_scalar("Train/Original_Segmentation_Loss",
                                  orig_seg_loss.item(), n_iter)
                # predict_and_visualize_heatmap(student_model,
                #                               sample_batched["image"], device,
                #                               save_root)
                n_iter += 1

            # 每個 epoch 結束後更新學習率並保存模型
            scheduler.step()
            torch.save(student_model.state_dict(),
                       os.path.join(checkpoint_dir, obj_name + ".pckl"))

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
    parser.add_argument('--gpu_id',
                        action='store',
                        type=int,
                        default=-2,
                        required=False,
                        help='GPU ID (-2: auto-select, -1: CPU)')
    args = parser.parse_args()

    # 自動選擇GPU
    if args.gpu_id == -2:  # 自動選擇模式
        args.gpu_id = get_available_gpu()
        print(f"自動選擇 GPU: {args.gpu_id}")

    obj_batch = [['capsule'], ['bottle'], ['carpet'], ['leather'], ['pill'],
                 ['transistor'], ['tile'], ['cable'], ['zipper'],
                 ['toothbrush'], ['metal_nut'], ['hazelnut'], ['screw'],
                 ['grid'], ['wood']]

    if int(args.obj_id) == -1:
        obj_list = [
            'capsule', 'bottle', 'carpet', 'leather', 'pill', 'transistor',
            'tile', 'cable', 'zipper', 'toothbrush', 'metal_nut', 'hazelnut',
            'screw', 'grid', 'wood'
        ]
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
