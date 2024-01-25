# Test the baseline model (use the middle one to predict)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset import SeqDataset
from model import MLP

def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)

    :param
    - y_true: real values
    - y_pred: predicted values

    :return
    - SMAPE value
    """
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    denominator[denominator == 0] = 1e-10
    smape_val = torch.mean((numerator / denominator)) * 100.0
    return smape_val.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

hidden_size_1 = 2048
hidden_size_2 = 2048
batch_size = 64
lr = 0.0001
epochs = 10
length_origin = 180
length_pre = 30

test_dataset = SeqDataset(length=length_origin + length_pre, split=0.8, behind=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

loss_function = nn.L1Loss().to(device)

total_loss = 0
total_smape = 0
with torch.no_grad():
    for sequence in test_data_loader:
        sequence = sequence.to(device)
        inputs = sequence[:, :length_origin]
        ground_truth = sequence[:, length_origin:]
        outputs, _ = torch.median(inputs, dim=1)
        outputs = outputs.unsqueeze(1).repeat(1, length_pre)
        loss = loss_function(outputs, ground_truth)
        total_loss += loss.item() * inputs.shape[0]

        with torch.no_grad():
            smape_value = smape(outputs, ground_truth)
            total_smape += smape_value * inputs.shape[0]
            print('smape', smape_value)
        print('loss', loss.item())

    average_loss = total_loss / len(test_dataset)
    print(f"Loss: {average_loss}")
    average_smape = total_smape / len(test_dataset)
    print(f"SMAPE: {average_smape}")
