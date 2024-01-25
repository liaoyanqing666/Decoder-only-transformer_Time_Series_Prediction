import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, file_dir='E:\大学本科课程\大三2\机器学习基础\期末\代码\时序数据回归预测',
                 file_name='train_weekday.csv', length=None,
                 split: float = None, behind=False):
        """
        :param len: The length of the sequence
        :param split: Whether to split the dataset into train and test
        :param behind: Choose the data behind the split point or in front of it (When split is True)
        """
        file = os.path.join(file_dir, file_name)
        self.data_frame = pd.read_csv(file, header=0, index_col=0)
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        # Random shuffling to ensure that the data is not similar at the time of splitting
        self.data_frame.columns = pd.to_datetime(self.data_frame.columns, format='%Y-%m-%d')
        if length is not None:
            self.data_frame = self.data_frame.iloc[:, -length:]
        if split is not None:
            num = int(split * len(self.data_frame))
            if behind:
                self.data_frame = self.data_frame.iloc[:num, :]
            else:
                self.data_frame = self.data_frame.iloc[num:, :]


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = self.data_frame.iloc[idx, :].values

        return torch.tensor(sample, dtype=torch.float32)
        # [seq_len]

if __name__ == '__main__':
    dataset = SeqDataset(length=100, split=0.8)

    sample = dataset[0]
    print("Sample:", sample)
