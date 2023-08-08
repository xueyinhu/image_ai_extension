import copy
import os

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets

from tqdm import tqdm
from net import MyNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = MyNet().to(device)
input_tensor = torch.randn(1, 3, 960, 320)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}

data_dir = r'F:\stack_torch_src\\'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(net.parameters())
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


def say_some(some):
    if type(some) is list:
        helper = ''
        for some_item in some:
            helper += '- ' + str(some_item) + ' -'
        some = helper
    print('-' * 10 + ' ' + str(some) + ' ' + '-' * 10)


best_acc = 0.0
best_model_wts = copy.deepcopy(net.state_dict())
for epoch in range(50):
    say_some(epoch + 1)

    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    for phase in ['train', 'val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(data_loaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer_ft.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                _, predicts = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer_ft.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicts == labels.data)
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        if phase == 'train':
            exp_lr_scheduler.step()
            train_loss, train_acc = epoch_loss, epoch_acc
        else:
            val_loss, val_acc = epoch_loss, epoch_acc
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
    torch.save(net, 'models/model_epoch_{}_train_{}_{}_val_{}_{}.pkl'.
               format(epoch + 1, train_acc, train_loss, val_acc, val_loss))
print('Best val Acc: {:4f}'.format(best_acc))
net.load_state_dict(best_model_wts)


