import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from dataset import OmniglotDataset, OmniglotTestDataset

from tqdm import tqdm

train_path = 'dataset\images_training'
test_path  = 'dataset\images_evaluation'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
test_batch_size = 20
lr = 6e-4
step_size = 500
gamma = 0.9
epochs = 100


train_transform = transforms.Compose([
    transforms.RandomAffine(15),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

class OmniglotModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # input = 1, 105, 105
        features = 4096
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),    # 64 96 96
            nn.ReLU(),
            nn.MaxPool2d(2),         # 64 48 48
            
            nn.Conv2d(64, 128, 7),   # 128 42 42
            nn.ReLU(),
            nn.MaxPool2d(2),         # 128 21 21
            
            nn.Conv2d(128, 128, 4),  # 128 18 18
            nn.ReLU(),
            nn.MaxPool2d(2),         # 128 9 9
            
            nn.Conv2d(128, 256, 4),  # 256 6 6
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, features),
            nn.Sigmoid()
        )
        
        self.output = nn.Sequential(
            nn.Linear(features, 1),
            nn.Sigmoid()
        )
        
        
    def forward_x(self, x):
        x = self.conv(x)
        x = self.fc(x)
        
        return x
        
    def forward(self, x1, x2):
        x1 = self.forward_x(x1)    
        x2 = self.forward_x(x2)    
        
        l1 = torch.abs(x1- x2)
        
        out = self.output(l1)
        
        return out

train_dataset = OmniglotDataset(train_path, transform=train_transform)

test_dataset = OmniglotTestDataset(test_path, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

model = OmniglotModel()
model = model.to(device)

ud = []
lossi = []

# Use this if you have trained the model
# model.load_state_dict(torch.load('OmniglotModel.pth'))

def accuracy():
    model.eval()
    with torch.no_grad():
        count = 0.0
        total = 0.0
        for idx, (im1, im2) in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            output = model(im1, im2)
            
            output = output.view(-1)
            
            pred = F.softmax(output)
            
            if(torch.argmax(pred).item() == 0):
                count += 1.0
            
            total +=1.0
    return count / total

loss_fn = nn.BCELoss(reduction='mean')

optim = AdamW(model.parameters(), lr=lr)
scheduler = StepLR(optim, step_size=step_size, gamma=gamma)

for ep in range(epochs):
    model.train()
    
    batch_loss = 0.0
    
    for image1, image2, labels in tqdm(train_dataloader, total = len(train_dataloader)):

    
        logits = model(image1, image2)
        
        optim.zero_grad()
        loss = loss_fn(logits, labels)
        
        loss.backward()
        optim.step()
        scheduler.step()
        
        with torch.no_grad():
            lossi.append(loss.log10().item())
            ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in model.parameters()])

            batch_loss += loss.item()
            
            
    print('Epoch loss: ', batch_loss/batch_size, 'Learning rate: ', scheduler.get_last_lr()[-1])    


acc = accuracy()
print('Test accuracy :', acc)

def classify_single_image(n_way=20):
    test_dataset = OmniglotTestDataset(test_path, transform=test_transform, way=n_way)
    test_loader = DataLoader(test_dataset, batch_size=n_way, shuffle=False)

    m1, m2 = next(iter(test_loader))

    model.eval()

    pred = model(m1, m2)
    pred = pred.cpu().detach().numpy().flatten()
    
    im1 = m1.cpu().numpy()
    im2 = m2.cpu().numpy()

    plt.figure(figsize=(40, 4))
    for i in range(n_way * 2):
        idx = i % n_way
        ax = plt.subplot(2, n_way, i+1)
        if i<n_way:
            ax.imshow(im1[idx].squeeze(), cmap='gray')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            max_index = np.argmax(pred)
            
            ax.imshow(im2[idx].squeeze(), cmap='gray' if idx != max_index else 'copper')
            ax.set_xlabel(f'{pred[idx]:.2f}', fontsize=24)
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            
classify_single_image()

