"""
Step1 数据加载模块
"""

import torch
from torch.utils.data import Dataset
import os
import numpy as np

class CloverClusterDataset(Dataset):
    """Clover聚类数据集"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        print(f"📁 Step1数据集加载: {len(self.file_list)} 个文件")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        
        reads = torch.FloatTensor(data['reads'])
        reference = torch.FloatTensor(data['reference'])
        
        return reads, reference
