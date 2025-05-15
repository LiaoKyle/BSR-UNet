"""
train_bsr_unet_with_ssim_psnr.py

在 BSR-UNet/dataset 下：
  - data_dirty_png: 2500 张输入脏图 (galaxy_image_{idx}_dirty.png)
  - data_moxing_png: 2500 张目标干净图 (galaxy_image_{idx}.png)

功能：
1. 构建成对路径并 8:2 划分 train/val
2. DirtyCleanDataset 加载（避免映射错误）
3. 定义 BSR-UNet
4. 定义 CombinedLoss (MSE + (1-SSIM) - PSNR)
5. 训练 & 验证循环，实时显示 loss、SSIM、PSNR
"""

import os, glob, re, random
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -----------------------------
# 1. 配置 & 数据集划分
# -----------------------------
BASE_DIR    = os.path.join(os.getcwd(), "BSR-UNet", "dataset")
DIRTY_DIR   = os.path.join(BASE_DIR, "data_dirty_png")
CLEAN_DIR   = os.path.join(BASE_DIR, "data_moxing_png")
SEED        = 42
TRAIN_RATIO = 0.8
BATCH_SIZE  = 8
NUM_EPOCHS  = 50
LR          = 1e-4
ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2  # MSE, SSIM, PSNR 权重

# 提取编号映射
def build_map(paths, pattern):
    regex = re.compile(pattern)
    mp = {}
    for p in paths:
        m = regex.search(os.path.basename(p))
        if m: mp[int(m.group(1))] = p
    return mp

dirty_map = build_map(glob.glob(os.path.join(DIRTY_DIR,"*.png")),
                      r'galaxy_image_(\d+)_dirty')
clean_map = build_map(glob.glob(os.path.join(CLEAN_DIR,"*.png")),
                      r'galaxy_image_(\d+)')
idxs = sorted(set(dirty_map)&set(clean_map))
pairs = [(dirty_map[i], clean_map[i]) for i in idxs]

random.seed(SEED)
random.shuffle(pairs)
n_train = int(len(pairs)*TRAIN_RATIO)
train_pairs = pairs[:n_train]
val_pairs   = pairs[n_train:]
print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

# -----------------------------
# 2. Dataset & DataLoader
# -----------------------------
class DirtyCleanDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.tf = transform or transforms.ToTensor()
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        dpath, cpath = self.pairs[idx]
        img_d = Image.open(dpath).convert("RGB")
        img_c = Image.open(cpath).convert("RGB")
        return self.tf(img_d), self.tf(img_c)

tf = transforms.ToTensor()  # 归一化到 [0,1]
train_loader = DataLoader(DirtyCleanDataset(train_pairs, tf),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(DirtyCleanDataset(val_pairs, tf),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

# -----------------------------
# 3. BSR‑UNet 模型定义
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch,ch,3,padding=1)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch,ch,3,padding=1)
    def forward(self,x):
        r = x
        x = self.act(self.conv1(x))
        x = self.conv2(x) + r
        return self.act(x)

class BSRUNet(nn.Module):
    def __init__(self,in_ch=3,base=64):
        super().__init__()
        # 编码器
        self.e1 = nn.Sequential(nn.Conv2d(in_ch,base,3,padding=1), ResBlock(base))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(nn.Conv2d(base,base*2,3,padding=1), ResBlock(base*2))
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(nn.Conv2d(base*2,base*4,3,padding=1), ResBlock(base*4))
        self.p3 = nn.MaxPool2d(2)
        self.e4 = nn.Sequential(nn.Conv2d(base*4,base*8,3,padding=1), ResBlock(base*8))
        self.p4 = nn.MaxPool2d(2)
        # 瓶颈
        self.b  = nn.Sequential(nn.Conv2d(base*8,base*16,3,padding=1),
                                ResBlock(base*16), ResBlock(base*16))
        # 解码器
        self.up4= nn.ConvTranspose2d(base*16,base*8,2,stride=2)
        self.d4 = nn.Sequential(nn.Conv2d(base*8*2,base*8,3,padding=1), ResBlock(base*8))
        self.up3= nn.ConvTranspose2d(base*8,base*4,2,stride=2)
        self.d3 = nn.Sequential(nn.Conv2d(base*4*2,base*4,3,padding=1), ResBlock(base*4))
        self.up2= nn.ConvTranspose2d(base*4,base*2,2,stride=2)
        self.d2 = nn.Sequential(nn.Conv2d(base*2*2,base*2,3,padding=1), ResBlock(base*2))
        self.up1= nn.ConvTranspose2d(base*2,base,2,stride=2)
        self.d1 = nn.Sequential(nn.Conv2d(base*2,base,3,padding=1), ResBlock(base))
        self.out= nn.Conv2d(base,in_ch,1)

    def forward(self,x):
        e1 = self.e1(x); p1=self.p1(e1)
        e2 = self.e2(p1); p2=self.p2(e2)
        e3 = self.e3(p2); p3=self.p3(e3)
        e4 = self.e4(p3); p4=self.p4(e4)
        b  = self.b(p4)
        d4 = torch.cat([self.up4(b),e4],1); d4=self.d4(d4)
        d3 = torch.cat([self.up3(d4),e3],1); d3=self.d3(d3)
        d2 = torch.cat([self.up2(d3),e2],1); d2=self.d2(d2)
        d1 = torch.cat([self.up1(d2),e1],1); d1=self.d1(d1)
        return self.out(d1)

# -----------------------------
# 4. SSIM 和 PSNR 函数
# -----------------------------
def create_window(ws, ch, device, dtype):
    gauss = torch.exp(-((torch.arange(ws)-ws//2)**2)/(2*1.5**2))
    gauss /= gauss.sum()
    kernel = gauss[:,None] @ gauss[None,:]
    w = kernel.expand(ch,1,ws,ws).to(device=device,dtype=dtype)
    return w

def ssim(img1, img2, data_range=1.0):
    # 简化版 SSIM
    device, dtype = img1.device, img1.dtype
    ch = img1.size(1); ws=11; pad=ws//2
    w = create_window(ws, ch, device, dtype)
    mu1 = F.conv2d(img1, w, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, w, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, w, padding=pad, groups=ch) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, w, padding=pad, groups=ch) - mu2_sq
    sigma12   = F.conv2d(img1*img2, w, padding=pad, groups=ch) - mu1_mu2
    C1 = (0.01*data_range)**2; C2=(0.03*data_range)**2
    s = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return s.mean()

def psnr(img1, img2, data_range=1.0):
    mse = F.mse_loss(img1, img2, reduction='none').view(img1.size(0),-1).mean(1)
    return (10*torch.log10(data_range**2/(mse+1e-8))).mean()

# -----------------------------
# 5. 训练循环
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BSRUNet().to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, NUM_EPOCHS+1):
    # 训练阶段
    model.train()
    t_loss = t_ssim = t_psnr = 0.0

    # 使用 enumerate(start=1) 代替 bar.n
    train_bar = tqdm(train_loader, desc=f"[Train E{epoch}/{NUM_EPOCHS}]")
    for i, (x, y) in enumerate(train_bar, start=1):
        x, y = x.to(device), y.to(device)
        dr = 1.0  # ToTensor() 已归一化到 [0,1]

        out = model(x)
        mse = F.mse_loss(out, y)
        s   = ssim(out, y, dr)   # 自定义 SSIM 函数
        p   = psnr(out, y, dr)   # 自定义 PSNR 函数
        loss= ALPHA*mse + BETA*(1-s) - GAMMA*p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加
        t_loss += loss.item()
        t_ssim += s.item()
        t_psnr += p.item()

        # 平均值计算中使用 i 而不是 bar.n，避免除零
        train_bar.set_postfix({
            "loss": f"{t_loss/i:.4f}",
            "SSIM": f"{t_ssim/i:.4f}",
            "PSNR": f"{t_psnr/i:.2f}"
        })

    # 验证阶段同理
    model.eval()
    v_loss = v_ssim = v_psnr = 0.0
    val_bar = tqdm(val_loader, desc=f"[ Val  E{epoch}/{NUM_EPOCHS}]")
    with torch.no_grad():
        for i, (x, y) in enumerate(val_bar, start=1):
            x, y = x.to(device), y.to(device)
            dr = 1.0

            out = model(x)
            mse = F.mse_loss(out, y)
            s   = ssim(out, y, dr)
            p   = psnr(out, y, dr)
            loss= ALPHA*mse + BETA*(1-s) - GAMMA*p

            v_loss += loss.item()
            v_ssim += s.item()
            v_psnr += p.item()

            val_bar.set_postfix({
                "loss": f"{v_loss/i:.4f}",
                "SSIM": f"{v_ssim/i:.4f}",
                "PSNR": f"{v_psnr/i:.2f}"
            })

    print(f"Epoch {epoch}: Train Loss {t_loss/bar.n:.4f}, SSIM {t_ssim/bar.n:.4f}, PSNR {t_psnr/bar.n:.2f} | "
          f"Val Loss {v_loss/bar.n:.4f}, SSIM {v_ssim/bar.n:.4f}, PSNR {v_psnr/bar.n:.2f}")

# 保存最优模型
torch.save(model.state_dict(), "best_bsr_unet.pth")
