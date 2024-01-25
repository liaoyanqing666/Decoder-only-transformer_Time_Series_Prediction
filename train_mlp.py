# Train the MLP model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset import SeqDataset
from model import MLP
from loss import SMAPELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


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

hidden_size_1 = 512
hidden_size_2 = 512
batch_size = 64
lr = 0.0001
epochs = 50
length_origin = 180
length_pre = 30

train_dataset = SeqDataset(length=length_origin + length_pre, split=0.8, behind=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = SeqDataset(length=length_origin + length_pre, split=0.8, behind=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(len(train_dataset))
print(len(test_dataset))

model = MLP(length_origin, hidden_size_1, hidden_size_2, length_pre).to(device)

loss_function = SMAPELoss().to(device)
optimizer = Adam(model.parameters(), lr=lr)

min_loss = 1e10
for epoch in range(epochs):
    total_loss = 0

    model.train()
    for i, sequence in enumerate(train_data_loader):
        sequence = sequence.to(device)
        inputs = sequence[:, :length_origin]
        ground_truth = sequence[:, length_origin:]

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.shape[0]

        if i % 100 == 0:
            print('loss', loss.item())

    average_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}, Training Loss: {average_loss}")

    # Testing phase
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for i, test_sequence in enumerate(test_data_loader):
            test_sequence = test_sequence.to(device)
            test_inputs = test_sequence[:, :length_origin]
            test_ground_truth = test_sequence[:, length_origin:]

            test_outputs = model(test_inputs)
            test_loss += loss_function(test_outputs, test_ground_truth).item() * test_inputs.shape[0]

    average_test_loss = test_loss / len(test_dataset)
    print(f"Epoch {epoch+1}, Testing Loss: {average_test_loss}")

    if average_test_loss < min_loss:
        min_loss = average_test_loss
        torch.save(model.state_dict(), 'mlp_model_parameter.pth')
    print(min_loss)
