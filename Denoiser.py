import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader

X_train_np = np.load('X_train1000.npy')
Y_train_np = np.load('Y_train1000.npy')

print(X_train_np[0].shape)

train_losses = []

noisy_tensor = torch.tensor(X_train_np)
clean_tensor = torch.tensor(Y_train_np)

print(noisy_tensor[0].shape)


class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=5, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1),
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1),
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=1),
            nn.ConvTranspose1d(128, 1, kernel_size=3, stride=1, padding=1)

        )

    def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


class CustomDataset(Dataset):
    def __init__(self, noisy_data, clean_data):
        self.noisy_data = noisy_data
        self.clean_data = clean_data

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        return {'noisy': self.noisy_data[idx], 'clean': self.clean_data[idx]}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DenoiseCNN()
model.to(device)

train_dataset = CustomDataset(noisy_data=noisy_tensor, clean_data=clean_tensor)

#optimizer = optim.RMSprop(model.parameters(), lr=0.000004, alpha=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000002)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.3)

criterion = nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

min_loss = float('inf')
best_epoch = 0

# Eğitim döngüsü
num_epochs = 1000
train_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data in train_loader:
        inputs, targets = data['noisy'].to(device), data['clean'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")


    if epoch_loss < min_loss:
        min_loss = epoch_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')

print(f"Best model found at epoch {best_epoch + 1} with loss: {min_loss}")

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
torch.save(model.state_dict(), 'model1000.pth')
