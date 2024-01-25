# Train Transformer model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from loss import SMAPELoss
from dataset import SeqDataset
from model import TransformerDecoderPredictor

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

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embed_size = 128
heads = 4
num_layers = 4
ff_hidden_dim = 384
batch_size = 256
lr = 0.0002
epochs = 100
length_origin = 180
length_pre = 30
length = length_origin + length_pre


train_dataset = SeqDataset(length=length, split=0.8, behind=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = SeqDataset(length=length, split=0.8, behind=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(len(train_dataset))
print(len(test_dataset))

model = TransformerDecoderPredictor(embed_size, heads, ff_hidden_dim, num_layers, dropout=0.3).to(device)

loss_function = SMAPELoss().to(device)
optimizer = Adam(model.parameters(), lr=lr)

mask = torch.triu(torch.ones(length-1, length-1), diagonal=1).to(device)
min_loss = 1e10
min_loss_2 = 1e10
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for i, inputs in enumerate(train_data_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()

        outputs = model(inputs[:, :-1], mask, use_rope=True)
        loss = loss_function(outputs, inputs[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.shape[0]

        if i % 100 == 0:
            print('Train loss', loss.item())

    average_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}, Training Loss: {average_loss}")

    # Testing loop
    model.eval()

    # Use the built-in predict function for testing (not recommended)
    # test_loss = 0
    # with torch.no_grad():
    #     for i, test_inputs in enumerate(test_data_loader):
    #         test_inputs = test_inputs.to(device)
    #
    #         test_outputs = model.predict(test_inputs[:, :length_origin], length_pre, use_kv_cache=False, use_rope=True)
    #         test_loss += loss_function(test_outputs, test_inputs[:, length_origin:]).item() * test_inputs.shape[0]
    #
    #         if i % 100 == 0:
    #             print('Test Loss', loss_function(test_outputs, test_inputs[:, length_origin:]).item())
    #
    # average_test_loss = test_loss / len(test_dataset)
    # print(f"Epoch {epoch+1}, Test Loss1: {average_test_loss}")
    #
    # if average_test_loss < min_loss:
    #     min_loss = average_test_loss
    #     torch.save(model.state_dict(), 'transformer_model_parameter.pth')
    # print(min_loss)

    test_loss = 0
    with torch.no_grad():
        for i, test_inputs in enumerate(test_data_loader):
            test_inputs = test_inputs.to(device)
            outputs = test_inputs[:, :length_origin]
            for j in range(length_pre):
                mask = torch.triu(torch.ones(outputs.shape[1], outputs.shape[1]), diagonal=1).to(device)
                test_outputs = model(outputs, mask, use_rope=True)
                outputs = torch.cat((outputs, test_outputs[:, -1:]), dim=1)
            test_loss += loss_function(outputs[:, length_origin:], test_inputs[:, length_origin:]).item() * test_inputs.shape[0]

            if i % 100 == 0:
                print('Test Loss', loss_function(outputs[:, length_origin:], test_inputs[:, length_origin:]).item())

    average_test_loss = test_loss / len(test_dataset)
    print(f"Epoch {epoch+1}, Test Loss: {average_test_loss}")
    if average_test_loss < min_loss_2:
        min_loss_2 = average_test_loss
        torch.save(model.state_dict(), 'transformer_model_parameter.pth')
    print(min_loss_2)
