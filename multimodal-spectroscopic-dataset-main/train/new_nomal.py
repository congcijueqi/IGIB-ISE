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

# Define CNN Model in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class IndependentCNN(nn.Module):
    def __init__(self, num_fgs):
        super(IndependentCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=31, kernel_size=11, padding='same')
        self.conv2 = nn.Conv1d(in_channels=31, out_channels=62, kernel_size=11, padding='same')

        self.batch_norm1 = nn.BatchNorm1d(31)
        self.batch_norm2 = nn.BatchNorm1d(62)

        # MLP for selecting important channels (62 channels)
        self.mlp = nn.Sequential(
            nn.Linear(150, 128),  # Input 150 features per channel
            nn.ReLU(),
            nn.Linear(128, 1)     # Output importance score for each channel
        )

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        # 通道重要性计算
        static_feature_map = x.clone().detach()
        channel_means = x.mean(dim=1)
        channel_std = x.std(dim=1)

        channel_importance = torch.sigmoid(self.mlp(x))
        ib_x_mean = x * channel_importance + (1 - channel_importance) * channel_means.unsqueeze(1)
        ib_x_std = (1 - channel_importance) * channel_std.unsqueeze(1)
        ib_x = ib_x_mean + torch.rand_like(ib_x_mean) * ib_x_std

        # KL Divergence loss
        epsilon = 1e-8
        KL_tensor = 0.5 * (
            (ib_x_std**2) / (channel_std.unsqueeze(1) + epsilon)**2 +
            (channel_std.unsqueeze(1)**2) / (ib_x_std + epsilon)**2 - 1
        ) + ((ib_x_mean - channel_means.unsqueeze(1))**2) / (channel_std.unsqueeze(1) + epsilon)**2

        KL_Loss = torch.mean(KL_tensor)

        # Flatten and pass through fully connected layers

        return ib_x, KL_Loss

class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        q = self.query(x)  # [batch_size, seq_len, input_dim]
        k = self.key(x)    # [batch_size, seq_len, input_dim]
        v = self.value(x)  # [batch_size, seq_len, input_dim]
        # 计算注意力分数
        attention_scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_probs = self.softmax(attention_scores)   # [batch_size, seq_len, seq_len]
        # 加权求和
        output = torch.bmm(attention_probs, v)  # [batch_size, seq_len, input_dim]
        return output


class CNNModel(nn.Module): 
    def __init__(self, num_fgs):
        super(CNNModel, self).__init__()
        # 创建三个独立的CNN模块
        self.cnn1 = IndependentCNN(num_fgs)
        self.cnn2 = IndependentCNN(num_fgs)
        self.cnn3 = IndependentCNN(num_fgs)

        # 自注意力层，用于信息交互
        self.attention = SelfAttentionLayer(input_dim=150)

        # 全连接层
        self.fc1 = nn.Linear(62 * 150*3, 4927)
        self.fc2 = nn.Linear(4927, 2785)
        self.fc3 = nn.Linear(2785, 1574)
        self.fc4 = nn.Linear(1574, num_fgs)
        self.dropout = nn.Dropout(0.48599073736368)

    def forward(self, x):
        # 拆分输入为三个通道
        x1, x2, x3 = x[:, 0:1, :], x[:, 1:2, :], x[:, 2:3, :]

        # 分别通过三个独立的CNN
        ib_x_1, kl1 = self.cnn1(x1)
        ib_x_2, kl2 = self.cnn2(x2)
        ib_x_3, kl3 = self.cnn3(x3)

        # 整合三个通道的输出
        # 将三个通道堆叠后通过自注意力机制增强交互
        ib_x_stacked = torch.concat([ib_x_1, ib_x_2, ib_x_3], dim=1)  # [batch_size, 3, seq_len, hidden_dim]
        ib_x_interacted = self.attention(ib_x_stacked)  # [batch_size, seq_len, hidden_dim]

        # 继续进行预测
        x = ib_x_interacted.view(ib_x_interacted.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))

        # KL损失取平均值
        kl_loss = (kl1 + kl2 + kl3) / 3
        return x, kl_loss



# Training function in PyTorch
from tqdm import tqdm  # 引入 tqdm

b=0.0001
def train_model(X_train, y_train, X_test,y_test, num_fgs, weighted=False, batch_size=41, epochs=41):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(num_fgs).to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    
    if weighted:
        class_weights = calculate_class_weights(y_train)
        criterion = WeightedBinaryCrossEntropyLoss(class_weights).to(device)
    else:
        criterion = nn.BCELoss().to(device)

    # Create DataLoader
    y_train = np.array([np.array(item, dtype=np.float32) for item in y_train], dtype=np.float32)
    y_test = np.array([np.array(item, dtype=np.float32) for item in y_test], dtype=np.float32)
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Train the model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Create tqdm progress bar for each epoch
        with tqdm(train_loader, unit='batch', desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for inputs, targets in tepoch:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs,loss1 = model(inputs)  # Add channel dimension
                loss2 = criterion(outputs, targets)
                loss = loss2+loss1*b
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # Update the progress bar with loss information
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))
        
        # After every epoch, print the average loss
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}')

    # Evaluate the model
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs,loss2 = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions)
    return (predictions > 0.5).astype(int)


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
    all_data.append(data)
    print(f"Loaded Data from: ", i)
    
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
print(y_train[0])
predictions = train_model(X_train, y_train, X_test, y_test,num_fgs=37, weighted=False)

# Evaluate the model
y_test = np.array([np.array(item, dtype=np.float32) for item in y_test], dtype=np.float32)
f1 = f1_score(y_test, predictions, average='micro')
print(f'F1 Score: {f1}')

# Save results
with open(out_path / "results.pickle", "wb") as file:
    pickle.dump({'pred': predictions, 'tgt': y_test}, file)