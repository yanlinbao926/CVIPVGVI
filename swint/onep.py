import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import logging
from torch.amp import autocast, GradScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

print("🔥 [单张基准模型] 当前模式: Swin-T 单视角输入 + HuberLoss免疫异常值 + 批次16")
CONFIG = {
    "train_csv_path": "", # Training set address
    "test_csv_path": "", # Test set address 
    "save_dir": "", # Output address
    "bottleneck_dim": 768,   
    "freeze_backbone": False, 
    "lr_backbone": 1e-5, 
    "lr_head": 1e-3,
    "wd_head": 0,            
    "wd_backbone": 0,        
    "batch_size": 16,        
    "epochs": 80, 
    "patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8
}

if not os.path.exists(CONFIG["save_dir"]): 
    os.makedirs(CONFIG["save_dir"])

log_file = os.path.join(CONFIG["save_dir"], "training.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    force=True, 
    handlers=[
        logging.FileHandler(log_file, mode='w'), 
        logging.StreamHandler()
    ]
)
logging.info(f"🚀 实验配置: {CONFIG}")
logging.info(f"🔓 随机种子: 42 (应用层锁定, 底层cuDNN基准测试开启)")

class SingleViewDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path_col = 'img_path' if 'img_path' in row.index else 'front_path'
        img = Image.open(row[img_path_col]).convert('RGB')
        
        if self.transform: 
            img = self.transform(img)
            
        return img, torch.tensor(float(row['pedestrian_gvi']), dtype=torch.float32)

class FlexibleSwinSingleModel(nn.Module):
    def __init__(self, bottleneck_dim=768, freeze_backbone=False):
        super(FlexibleSwinSingleModel, self).__init__()
        
        logging.info("🔄 正在加载 PyTorch 官方 Swin-T 预训练权重...")
        self.backbone = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        
        self.backbone.head = nn.Identity()
        logging.info("✅ 官方预训练权重加载完毕")
            
        for param in self.backbone.parameters(): 
            param.requires_grad = not freeze_backbone
        logging.info("🔒 Swin-T Backbone 已冻结" if freeze_backbone else "🔓 Swin-T Backbone 已解冻")
            
        self.bottleneck = nn.Sequential(
            nn.Linear(768, bottleneck_dim), 
            nn.LayerNorm(bottleneck_dim), 
            nn.ReLU(),
            nn.Dropout(0.5) 
        )
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, 256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        feat = self.backbone(x)
        if len(feat.shape) > 2:
            feat = torch.flatten(feat, 1) 
        
        feat = self.bottleneck(feat)
        return self.head(feat)

def run_training():

    train_transform = transforms.Compose([
        transforms.Resize((256, 512)), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 512)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logging.info(f"📖 读取训练大沙盒数据: {CONFIG['train_csv_path']}")
    sandbox_df = pd.read_csv(CONFIG['train_csv_path'])
    if len(sandbox_df) == 0: raise ValueError("训练沙盒数据为空")

    logging.info(f"📖 读取独立期末测试集: {CONFIG['test_csv_path']}")
    test_df = pd.read_csv(CONFIG['test_csv_path'])
    if len(test_df) == 0: raise ValueError("测试集数据为空")

    train_df, val_df = train_test_split(sandbox_df, test_size=0.1, random_state=42)
    logging.info(f"📊 样本数划分: 训练集(Train) {len(train_df)} | 验证集(Val) {len(val_df)} | 测试集(Test) {len(test_df)}")

    train_loader = DataLoader(
        SingleViewDataset(train_df, train_transform), 
        batch_size=CONFIG["batch_size"], shuffle=True, 
        num_workers=CONFIG["num_workers"], drop_last=True,
        worker_init_fn=seed_worker, generator=g                 
    )
    
    val_loader = DataLoader(
        SingleViewDataset(val_df, val_transform), 
        batch_size=CONFIG["batch_size"], shuffle=False, 
        num_workers=CONFIG["num_workers"], drop_last=False
    )
    
    test_loader = DataLoader(
        SingleViewDataset(test_df, val_transform), 
        batch_size=CONFIG["batch_size"], shuffle=False, 
        num_workers=CONFIG["num_workers"], drop_last=False
    )
    
    model = FlexibleSwinSingleModel(
        bottleneck_dim=CONFIG["bottleneck_dim"],
        freeze_backbone=CONFIG["freeze_backbone"]
    ).to(CONFIG["device"])
    
    criterion = nn.HuberLoss(delta=0.1)
    
    params = [
        {'params': model.head.parameters(), 'lr': CONFIG["lr_head"], 'weight_decay': CONFIG["wd_head"]},
        {'params': model.bottleneck.parameters(), 'lr': CONFIG["lr_head"], 'weight_decay': CONFIG["wd_head"]}
    ]
    if not CONFIG["freeze_backbone"]:
        params.append({'params': model.backbone.parameters(), 'lr': CONFIG["lr_backbone"], 'weight_decay': CONFIG["wd_backbone"]})
        
    optimizer = optim.Adam(params) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    scaler = GradScaler('cuda') if CONFIG["device"] == "cuda" else None
    
    best_loss = float('inf')
    best_epoch = 0 
    early_stop = 0

    logging.info("🚀 开始训练...")
    for epoch in range(CONFIG["epochs"]):
        
        model.train()
        if CONFIG["freeze_backbone"]:
            model.backbone.eval()
            
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"]).unsqueeze(1)
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            train_loss += loss.item()
            
        model.eval()
        preds, trues = [],[]
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"]).unsqueeze(1)
                
                if scaler is not None:
                    with autocast('cuda'):
                        out = model(x)
                else:
                    out = model(x)
                    
                preds.extend(out.cpu().numpy().flatten())
                trues.extend(y.cpu().numpy().flatten())

        avg_train = train_loss / len(train_loader) 
        
        raw_val_loss = mean_squared_error(trues, preds) 
        
        preds_clipped = np.clip(preds, 0.0, 1.0)
        val_mse_clipped = mean_squared_error(trues, preds_clipped) 
        mae = mean_absolute_error(trues, preds_clipped)
        r2 = r2_score(trues, preds_clipped)
        
        if np.std(preds_clipped) == 0 or np.std(trues) == 0:
            pearson_r_val = 0.0
        else:
            pearson_r_val, _ = pearsonr(trues, preds_clipped)

        scheduler.step(raw_val_loss)
        
        lr_head = optimizer.param_groups[0]['lr']
        lr_str = f"Head:{lr_head:.2e}"
        if not CONFIG["freeze_backbone"] and len(optimizer.param_groups) > 1:
            lr_str += f", Backbone:{optimizer.param_groups[-1]['lr']:.2e}"
            
        log_msg = (f"Epoch {epoch+1:02d} | {lr_str} | Train HuberLoss={avg_train:.4f} | "
                   f"Val MSE={val_mse_clipped:.4f} | Val MAE={mae:.4f} | "
                   f"R²={r2:.4f} | Pearson's r={pearson_r_val:.4f}")
        logging.info(log_msg)

        if raw_val_loss < best_loss:
            best_loss = raw_val_loss
            best_epoch = epoch + 1 
            early_stop = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
            logging.info(f"  💾 [模型更新] 发现更低误差！已将 Epoch {best_epoch} 的权重保存为 best_model.pth")
        else:
            early_stop += 1
            if early_stop >= CONFIG["patience"]:
                logging.info(f"🛑 早停触发！模型原生误差已连续 {CONFIG['patience']} 轮未下降。")
                break

    logging.info("="*50)
    logging.info("🎉 训练阶段正式结束！")
    logging.info(f"🏆 最佳模型定格在第 {best_epoch} 轮。")
    logging.info("="*50)
    
    logging.info("🧪 正在加载最佳模型，对带隔离带的测试集(Test Set)进行终极评估...")
    
    model.load_state_dict(torch.load(os.path.join(CONFIG["save_dir"], "best_model.pth")))
    model.eval()
    
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x, y in test_loader: 
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"]).unsqueeze(1)

            if scaler is not None:
                with autocast('cuda'):
                    out = model(x)
            else:
                out = model(x)
                
            test_preds.extend(out.cpu().numpy().flatten())
            test_trues.extend(y.cpu().numpy().flatten())
            
    test_preds_clipped = np.clip(test_preds, 0.0, 1.0)
    test_mse = mean_squared_error(test_trues, test_preds_clipped) 
    test_mae = mean_absolute_error(test_trues, test_preds_clipped)
    test_r2 = r2_score(test_trues, test_preds_clipped)
    
    if np.std(test_preds_clipped) == 0 or np.std(test_trues) == 0:
        test_pearson = 0.0
    else:
        test_pearson, _ = pearsonr(test_trues, test_preds_clipped)

    logging.info("\n" + "="*45)
    logging.info(f"🏆 独立测试集终极评估成绩 (请将此成绩写入论文！):")
    logging.info(f"   ➤ MSE : {test_mse:.5f}")
    logging.info(f"   ➤ MAE : {test_mae:.5f}")
    logging.info(f"   ➤ R²  : {test_r2:.5f}")
    logging.info(f"   ➤ Pearson's r : {test_pearson:.5f}")
    logging.info("="*45 + "\n")

if __name__ == '__main__':
    run_training()