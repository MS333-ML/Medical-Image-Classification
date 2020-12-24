import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.decomposition import PCA
import torch.optim
import torch.utils.data as Data
import torch._C


root = 'datasets/'
npz_file = np.load(os.path.join(root, 'breastmnist.npz'))
train_data = npz_file['train_images']
train_labels =  npz_file['train_labels']
val_data = npz_file['val_images']
val_labels =  npz_file['val_labels']
test_data = npz_file['test_images']
test_labels =  npz_file['test_labels']

pca=PCA(n_components=20)
reduction_method = pca
Q_code=reduction_method.fit_transform(train_data.reshape(train_data.shape[0], -1))
test_code = reduction_method.fit_transform(test_data.reshape(test_data.shape[0], -1))

net = nn.Linear(pca.n_components, 2)
net = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(pca.n_components, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(20, 2)
)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=net.parameters())
epochs = 10000
dataset = Data.TensorDataset(torch.from_numpy(Q_code).float(), torch.from_numpy(train_labels).long())
dataloader = Data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

test_dataset = Data.TensorDataset(torch.from_numpy(test_code).float(), torch.from_numpy(test_labels).long())
test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=32)

def evaluate():
    net.eval()
    test_len = test_code.shape[0]
    correct_cnt = 0

    for step, (batchx, batchy) in enumerate(test_dataloader):
        pred = net(batchx)
        correct_cnt += torch.sum(torch.argmax(pred, dim=1) == batchy.squeeze()).item()

    print(f'Correct: {correct_cnt}/{test_len}')

for epoch in range(epochs):
    net.train()
    total_loss = 0.0
    steps = 0
    for step, (batchx, batchy) in enumerate(dataloader):
        pred = net(batchx)
        loss = loss_func(pred, batchy.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    if epoch % 50 == 0:
        print(f'Epoch: {epoch}, Loss: {total_loss/steps}')
        evaluate()
