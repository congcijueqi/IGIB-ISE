#论文中需要预测的部分，用 t3去预测y构建一个loss，  用t1，t2，t3预测y构建一个loss， t1和t2与 t3之间最小化构建一个loss
#  t3与x3之间最小化构建一个loss， t1与x3，x1最小化构建一个loss，t2与x3和x2最小化构建一个loss


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset

# Disable RDLogger warnings
RDLogger.DisableLog('rdApp.*')
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
functional_groups = {
    'Acid anhydride': Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])'),
    'Acyl halide': Chem.MolFromSmarts('[CX3](=[OX1])[F,Cl,Br,I]'),
    'Alcohol': Chem.MolFromSmarts('[#6][OX2H]'),
    'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)[#6,H]'),
    'Alkane': Chem.MolFromSmarts('[CX4;H3,H2]'),
    'Alkene': Chem.MolFromSmarts('[CX3]=[CX3]'),
    'Alkyne': Chem.MolFromSmarts('[CX2]#[CX2]'),
    'Amide': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[#6]'),
    'Amine': Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]'),
    'Arene': Chem.MolFromSmarts('[cX3]1[cX3][cX3][cX3][cX3][cX3]1'),
    'Azo compound': Chem.MolFromSmarts('[#6][NX2]=[NX2][#6]'),
    'Carbamate': Chem.MolFromSmarts('[NX3][CX3](=[OX1])[OX2H0]'),
    'Carboxylic acid': Chem.MolFromSmarts('[CX3](=O)[OX2H]'),
    'Enamine': Chem.MolFromSmarts('[NX3][CX3]=[CX3]'),
    'Enol': Chem.MolFromSmarts('[OX2H][#6X3]=[#6]'),
    'Ester': Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]'),
    'Ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
    'Haloalkane': Chem.MolFromSmarts('[#6][F,Cl,Br,I]'),
    'Hydrazine': Chem.MolFromSmarts('[NX3][NX3]'),
    'Hydrazone': Chem.MolFromSmarts('[NX3][NX2]=[#6]'),
    'Imide': Chem.MolFromSmarts('[CX3](=[OX1])[NX3][CX3](=[OX1])'),
    'Imine': Chem.MolFromSmarts('[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]'),
    'Isocyanate': Chem.MolFromSmarts('[NX2]=[C]=[O]'),
    'Isothiocyanate': Chem.MolFromSmarts('[NX2]=[C]=[S]'),
    'Ketone': Chem.MolFromSmarts('[#6][CX3](=O)[#6]'),
    'Nitrile': Chem.MolFromSmarts('[NX1]#[CX2]'),
    'Phenol': Chem.MolFromSmarts('[OX2H][cX3]:[c]'),
    'Phosphine': Chem.MolFromSmarts('[PX3]'),
    'Sulfide': Chem.MolFromSmarts('[#16X2H0]'),
    'Sulfonamide': Chem.MolFromSmarts('[#16X4]([NX3])(=[OX1])(=[OX1])[#6]'),
    'Sulfonate': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]'),
    'Sulfone': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[#6]'),
    'Sulfonic acid': Chem.MolFromSmarts('[#16X4](=[OX1])(=[OX1])([#6])[OX2H]'),
    'Sulfoxide': Chem.MolFromSmarts('[#16X3]=[OX1]'),
    'Thial': Chem.MolFromSmarts('[CX3H1](=S)[#6,H]'),
    'Thioamide': Chem.MolFromSmarts('[NX3][CX3]=[SX1]'),
    'Thiol': Chem.MolFromSmarts('[#16X2H]')
}
def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) == Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1
# Function to map SMILES to functional groups (no change)
def get_functional_groups(smiles: str) -> dict:
    smiles = smiles.strip().replace(' ', '')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    func_groups = [match_group(mol, smarts) for smarts in functional_groups.values()]
    return func_groups

def interpolate_to_600(spec):
    old_x = np.arange(len(spec))
    new_x = np.linspace(min(old_x), max(old_x), 600)
    interp = interp1d(old_x, spec)
    return interp(new_x)

def make_msms_spectrum(spectrum):
    msms_spectrum = np.zeros(10000)
    for peak in spectrum:
        peak_pos = int(peak[0]*10)
        peak_pos = min(peak_pos, 9999)
        msms_spectrum[peak_pos] = peak[1]
    return msms_spectrum

# Define CNN Model in PyTo




import torch
import torch.nn as nn
import torch.nn.functional as F




#论文中需要预测的部分，用 t3去预测y构建一个loss，  用t1，t2，t3预测y构建一个loss， t1和t2与 t3之间最小化构建一个loss
#  t3与x3之间最小化构建一个loss， t1与x3，x1最小化构建一个loss，t2与x3和x2最小化构建一个loss

class IndependentCNN_main(nn.Module):
    def __init__(self, num_fgs):
        super(IndependentCNN_main, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=31, kernel_size=11, padding='same')
        self.conv2 = nn.Conv1d(in_channels=31, out_channels=62, kernel_size=11, padding='same')

        self.batch_norm1 = nn.BatchNorm1d(31)
        self.batch_norm2 = nn.BatchNorm1d(62)

        # MLP for selecting important channels (62 channels)
        self.mlp = nn.Sequential(
            nn.Linear(62, 128),  # 输入每个通道150个特征
            nn.ReLU(),
            nn.Linear(128, 1)     # 输出每个通道的重要性评分
        )

    def compress(self, solute_features):
        p = self.mlp(solute_features)
        device = solute_features.device
        temperature = 1.0
        bias = 0.0001  # 避免 bias 为 0 导致的问题
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()
        p = torch.sigmoid(p)
        return gate_inputs, p

    def forward(self, x1,x2,x3):
        
        
    
        
        x3 = F.relu(self.batch_norm1(self.conv1(x3)))
        x3 = F.max_pool1d(x3, 2)  # 池化大小为1，不改变尺寸
        x3 = F.relu(self.batch_norm2(self.conv2(x3)))
        x3 = F.max_pool1d(x3, 2)  # 池化大小为4，减少特征维度
        x3 = x3.permute(0, 2, 1)  # 调整维度顺序

        # 先计算出他们各自的方差：
        channel_std_3 = x3.std(dim=1)

        # 压缩与门控，依次压缩 1 2 3
        
    

        # **x1部分开始**其中t_m只由x_m决定
        x1 = F.relu(self.batch_norm1(self.conv1(x1)))
        x1 = F.max_pool1d(x1, 2)  # 池化大小为1，不改变尺寸
        x1 = F.relu(self.batch_norm2(self.conv2(x1)))
        x1 = F.max_pool1d(x1, 2)  # 池化大小为4，减少特征维度
        x1 = x1.permute(0, 2, 1)  # 调整维度顺序
        channel_std_1 = x1.std(dim=1)
        channel_importance_1, p_1 = self.compress(x1)
        channel_importance_1 = channel_importance_1.unsqueeze(-1)
        
        ib_x1_mean = x1 * channel_importance_1  # 去除 (1 - channel_importance) * channel_means.unsqueeze(1)
        ib_x1_std = (1 - channel_importance_1) * channel_std_1.unsqueeze(1)
        ib_x1 = ib_x1_mean + torch.rand_like(ib_x1_mean) * ib_x1_std
        
         # KL 散度损失计算,先算 ta tm 是ta与他的0 1做kl
        epsilon = 1e-8
        KL_tensor = 0.5 * (
            (ib_x1_std**2) / (channel_std_1.unsqueeze(1) + epsilon)**2 +
            (channel_std_1.unsqueeze(1)**2) / (ib_x1_std + epsilon)**2 - 1
        ) + (ib_x1_mean**2) / (channel_std_1.unsqueeze(1) + epsilon)**2  # 修改了这里，将 (ib_x_mean - 0)**2 替换为 ib_x_mean**2

        KL_Loss_1 = torch.mean(KL_tensor)
        # **修改部分结束**
        

        # **x2部分开始**，t_a是由另外三个x_m.t_m,x_a构成的
        x2 = F.relu(self.batch_norm1(self.conv1(x2)))
        x2 = F.max_pool1d(x2, 2)  # 池化大小为1，不改变尺寸
        x2 = F.relu(self.batch_norm2(self.conv2(x2)))

        x2 = F.max_pool1d(x2, 2)  # 池化大小为4，减少特征维度
        x2 = x2.permute(0, 2, 1)  # 调整维度顺序.
        print(x2.size())
        channel_std_2 = x2.std(dim=1)
        channel_importance_2, p_2 = self.compress(x2)
        channel_importance_2 = channel_importance_2.unsqueeze(-1)
        ib_x2_mean = x2 * channel_importance_2  # 去除 (1 - channel_importance) * channel_means.unsqueeze(1)
        ib_x2_std = (1 - channel_importance_2) * channel_std_2.unsqueeze(1)
        ib_x2 = ib_x2_mean + torch.rand_like(ib_x2_mean) * ib_x2_std
        
        epsilon = 1e-8
        KL_tensor = 0.5 * (
            (ib_x2_std**2) / (channel_std_2.unsqueeze(1) + epsilon)**2 +
            (channel_std_2.unsqueeze(1)**2) / (ib_x2_std + epsilon)**2 - 1
        ) + (ib_x2_mean**2) / (channel_std_2.unsqueeze(1) + epsilon)**2  # 修改了这里，将 (ib_x_mean - 0)**2 替换为 ib_x_mean**2

        KL_Loss_2 = torch.mean(KL_tensor)
        
        # **修改部分结束**
        
        channel_importance_3, p_3 = self.compress(x3)
        channel_importance_3 = channel_importance_3.unsqueeze(-1)

        # **x1部分开始**
        # 将不重要的部分置为0，而不是均值
        x3 = F.relu(self.batch_norm1(self.conv1(x3)))
        x3 = F.max_pool1d(x3, 2)  # 池化大小为1，不改变尺寸
        x3 = F.relu(self.batch_norm2(self.conv2(x3)))
        x3 = F.max_pool1d(x3, 2)  # 池化大小为4，减少特征维度
        x3 = x3.permute(0, 2, 1)  # 调整维度顺序

        # 先计算出他们各自的方差：
        channel_std_3 = x3.std(dim=1)
        channel_importance_3, p_3 = self.compress(x3)
        channel_importance_3 = channel_importance_3.unsqueeze(-1)
        ib_x3_mean = x3 * channel_importance_3  # 去除 (1 - channel_importance) * channel_means.unsqueeze(1)
        ib_x3_std = (1 - channel_importance_3) * channel_std_3.unsqueeze(1)
        ib_x3 = ib_x3_mean + torch.rand_like(ib_x3_mean) * ib_x3_std

    
        





        # 第二个KL 散度损失计算,先算 xm tm与
        epsilon = 1e-8
        KL_tensor = 0.5 * (
            (ib_x3_std**2) / (channel_std_3.unsqueeze(1) + epsilon)**2 +
            (channel_std_3.unsqueeze(1)**2) / (ib_x3_std + epsilon)**2 - 1
        ) + (ib_x1_mean**3) / (channel_std_3.unsqueeze(1) + epsilon)**2  # 修改了这里，将 (ib_x_mean - 0)**2 替换为 ib_x_mean**2

        KL_Loss_3 = torch.mean(KL_tensor)
        KL_Loss = KL_Loss_1+KL_Loss_2+KL_Loss_3
        
        ib_x1 = ib_x1.permute(0, 2, 1)  # 调整维度顺序
        ib_x2 = ib_x2.permute(0, 2, 1)  # 调整维度顺序
        ib_x3 = ib_x3.permute(0, 2, 1)  # 调整维度顺序
        return ib_x1,ib_x2,ib_x3, KL_Loss, p_1,p_2,p_3












import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModelWithVAE(nn.Module): 
    def __init__(self, num_fgs, channel=62, feature_dim=150, hidden_dim=256, latent_dim=64, m_dim=10):
        """
        参数：
        - num_fgs: 预测目标的维度
        - channel: 每个光谱的通道数（不同频率段）
        - feature_dim: 每个光谱的特征维度
        - hidden_dim: 隐藏层维度
        - latent_dim: 潜在变量 z 的维度
        - m_dim: 预测目标的维度（如有需要）
        """
        super(CNNModelWithVAE, self).__init__()
        self.channel = channel
        self.feature_dim = feature_dim

        # 创建三个独立的CNN模块
        self.cnn1 = IndependentCNN_main(num_fgs)
        self.mha = nn.MultiheadAttention(embed_dim=62,
                                num_heads=2,
                                batch_first=False)






        # 全连接层用于最终预测，使用 z 和 x3 作为输入
        self.fc1 = nn.Linear(channel *feature_dim, 4927)  # z 和 x3
        self.fc2 = nn.Linear(4927, 2785)
        self.fc3 = nn.Linear(2785, 1574)
        self.fc4 = nn.Linear(1574, num_fgs)
        self.dropout = nn.Dropout(0.48599073736368)
    
    
    def forward(self, x):
        """
        前向传播函数。
        
        参数：
        - x: 输入张量，形状为 [batch_size, 3, feature_dim]
        
        返回：
        - 一个包含预测结果和各类损失组件的字典
        """
        # 拆分输入为三个光谱通道
        x1, x2, x3 = x[:, 0:1, :], x[:, 1:2, :], x[:, 2:3, :]  # 每个 [B, 1, feature_dim]

        # 分别通过三个独立的CNN
        ib_x_1,ib_x_2,ib_x_3, KL_Loss, channal_importance_1,channal_importance_2,channal_importance_3= self.cnn1(x1,x2,x3)
    
        # 2. 调整输入形状: [B, C, F] -> [F, B, C]
        #    假设 feature_dim=F 是序列长度, channel=C 是嵌入维度
        ib_x_1_t = ib_x_1.permute(2, 0, 1)  # [F, B, C]
        ib_x_2_t = ib_x_2.permute(2, 0, 1)  # [F, B, C]
        ib_x_3_t = ib_x_3.permute(2, 0, 1)  # [F, B, C]

        # 3. 先以 ib_x_3_t 做 Query，ib_x_1_t 做 Key & Value
        out1, _ = self.mha(query=ib_x_3_t, key=ib_x_1_t, value=ib_x_1_t)
        # 4. 再以 ib_x_3_t 做 Query，ib_x_2_t 做 Key & Value
        out2, _ = self.mha(query=ib_x_3_t, key=ib_x_2_t, value=ib_x_2_t)

        # 5. 将两次输出与残差(ib_x_3_t)融合
        out = out1 + out2 + ib_x_3_t  # 残差连接，可根据需要再加 LayerNorm 等后处理

        # 6. 调整回原形状: [F, B, C] -> [B, C, F]
        z = out.permute(1, 2, 0)
        # VAE Decoder: 重建三个光谱
        x = ib_x_3.view(ib_x_3.size(0), -1)
        #现在需要z和x3分开处理，下面先突出x3的处理
        z = z.reshape(z.size(0), -1)
        ib_x_3= ib_x_3.view(ib_x_3.size(0), -1)
        x_pred = F.relu(self.fc1(ib_x_3))  # [B, 4927]
        x_pred = self.dropout(x_pred)
        x_pred = F.relu(self.fc2(x_pred))  # [B, 2785]
        x_pred = self.dropout(x_pred)
        x_pred = F.relu(self.fc3(x_pred))  # [B, 1574]
        x_pred = self.dropout(x_pred)
        x_pred_assit = torch.sigmoid(self.fc4(x_pred))  # [B, num_fgs]
        
        
        x_pred = F.relu(self.fc1(z ))  # [B, 4927]
        x_pred = self.dropout(x_pred)
        x_pred = F.relu(self.fc2(x_pred))  # [B, 2785]
        x_pred = self.dropout(x_pred)
        x_pred = F.relu(self.fc3(x_pred))  # [B, 1574]
        x_pred = self.dropout(x_pred)
        x_pred = torch.sigmoid(self.fc4(x_pred))  # [B, num_fgs]

        # KL散度损失取平均值（来自 VAE）
        kl=KL_Loss
        return {
            'x': x_pred,
            'ib_x_1': ib_x_1,
            'ib_x_2': ib_x_2,
            'ib_x_3': ib_x_3,
            'kl':kl,
            'channal_importance_1':channal_importance_1,
            'channal_importance_2':channal_importance_2,
            'channal_importance_3':channal_importance_3,
            'x_pred_assit':x_pred_assit
        }



# Training function in PyTorch
from tqdm import tqdm  # 引入 tqdm

b=0.0001
# 定义训练函数
# 定义训练函数
from tqdm import tqdm  # 引入 tqdm

# 定义训练函数
def train_model(X_train, y_train, X_test, y_test, num_fgs, weighted=False, batch_size=41, epochs=41, 
                annealing_epochs=10, max_lambda_kl=1.0, lambda_cmi=0.5, lambda_recon=0.0001):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = CNNModelWithVAE(num_fgs).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    if weighted:
        class_weights = calculate_class_weights(y_train)
        criterion = WeightedBinaryCrossEntropyLoss(class_weights).to(device)
    else:
        criterion = nn.BCELoss().to(device)

    # 创建 DataLoader
    y_train = np.array([np.array(item, dtype=np.float32) for item in y_train], dtype=np.float32)
    y_test = np.array([np.array(item, dtype=np.float32) for item in y_test], dtype=np.float32)
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 确保保存路径存在
    out_path.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        recon_loss_avg = 0.0
        kl_weight = min(max_lambda_kl, (epoch + 1) / annealing_epochs)
        with tqdm(train_loader, unit='batch', desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for batch in tepoch:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                x_pred = outputs['x']
                kl = outputs['kl']
                x_pred_assit = outputs['x_pred_assit']

                # 预测损失
                assit_loss = criterion(x_pred_assit, targets)
                pred_loss = criterion(x_pred, targets)
                
        

                # 总损失：预测损失 + KL散度 + 互信息损失 + 重建损失
                total_loss =pred_loss +   + 0.000000000001 * kl+assit_loss
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += total_loss.item()
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1),
                                  kl_weight=kl_weight)
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, KL Weight: {kl_weight}')
        
        # 评估F1分数
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                x_pred = outputs['x']
                predictions.append(x_pred.cpu().numpy())
        predictions = np.concatenate(predictions)
        binary_predictions = (predictions > 0.5).astype(int)
        f1 = f1_score(y_test, binary_predictions, average='micro')
        print(f'F1 Score: {f1}')
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            model_save_path = out_path / "best_model.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with F1 Score: {best_f1} at {model_save_path}')

    return binary_predictions




# Custom loss function with class weights
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        loss = self.class_weights[0] * (1 - y_true) * torch.log(1 - y_pred + 1e-15) + \
               self.class_weights[1] * y_true * torch.log(y_pred + 1e-15)
        return -loss.mean()

# Calculate class weights
def calculate_class_weights(y_true):
    num_samples = y_true.shape[0]
    class_weights = np.zeros((2, y_true.shape[1]))
    for i in range(y_true.shape[1]):
        weights_n = num_samples / (2 * (y_true[:, i] == 0).sum())
        weights_p = num_samples / (2 * (y_true[:, i] == 1).sum())
        class_weights[0, i] = weights_n
        class_weights[1, i] = weights_p
    return torch.tensor(class_weights.T, dtype=torch.float32)

# Loading data (no change)
analytical_data = Path("/data/zjh2/multimodal-spectroscopic-dataset-main/data/multimodal_spectroscopic_dataset")
out_path = Path("/home/dwj/icml_guangpu/multimodal-spectroscopic-dataset-main/runs/runs_f_groups/all")
columns = ["h_nmr_spectra", "c_nmr_spectra", "ir_spectra"]
seed = 3245

# 准备存储合并后的数据
all_data = []
i=0
# 一次性读取文件并处理所有列
for parquet_file in analytical_data.glob("*.parquet"):
    i+=1
    # 读取所有需要的列
    data = pd.read_parquet(parquet_file, columns=columns + ['smiles'])
    
    # 对每个列进行插值
    for column in columns:
        data[column] = data[column].map(interpolate_to_600)
    
    # 添加功能团信息
    data['func_group'] = data.smiles.map(get_functional_groups)
    #在这里就是0/1矩阵了
    all_data.append(data)
    print(f"Loaded Data from: ", i)
    if i== 3:
        break
# 合并所有数据
training_data = pd.concat(all_data, ignore_index=True)


# 将数据划分为训练集和测试集
train, test = train_test_split(training_data, test_size=0.1, random_state=seed)

# 定义特征列
columns = ["h_nmr_spectra", "c_nmr_spectra", "ir_spectra"]

# 提取训练集特征和标签
X_train = np.array(train[columns].values.tolist())  # 确保特征值是一个二维数组
y_train = np.array(train['func_group'].values)      # 标签转换为一维数组

# 提取测试集特征和标签
X_test = np.array(test[columns].values.tolist())    # 同样确保二维数组
y_test = np.array(test['func_group'].values)        # 标签一维数组

# 检查数组形状以验证正确性
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# Train extended model
predictions = train_model(X_train, y_train, X_test, y_test,num_fgs=37, weighted=False, batch_size=41, epochs=41, 
                annealing_epochs=10, max_lambda_kl=1.0, lambda_cmi=0.1, lambda_recon=0.1)

# Evaluate the model
y_test = np.array([np.array(item, dtype=np.float32) for item in y_test], dtype=np.float32)
f1 = f1_score(y_test, predictions, average='micro')
print(f'F1 Score: {f1}')

# Save results
with open(out_path / "results.pickle", "wb") as file:
    pickle.dump({'pred': predictions, 'tgt': y_test}, file)