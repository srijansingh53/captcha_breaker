import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import json

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,16,kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,kernel_size = 3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2))

        self.fc_model = nn.Sequential(
            nn.Linear(512, 256), # (N, 2592) -> (N, 512)
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(256, 28))  # (N, 512)  -> (N, 30)) #30 classes
            
    def forward(self, x):
        # print(x.shape)
        x = self.cnn_model(x)
        # print(x.shape) 
        x = x.view(x.size(0), -1)
        # print(x.shape)    
        x = self.fc_model(x)
        # print(x.shape)
        return x

def data_loader():
    data_dir = 'active letter'
    image_transforms = {
    "train": transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((100, 100)),
        transforms.ToTensor()
        ])
    }
    letters_dataset = datasets.ImageFolder(
                                root = data_dir,
                                transform = image_transforms["train"]
                        )
    print(letters_dataset)
    train_loader = DataLoader(dataset=letters_dataset, shuffle=True, batch_size=32)
    val_loader = DataLoader(dataset=letters_dataset, shuffle=False, batch_size=32)
    print("Length of the train_loader:", len(train_loader))
    print("Length of the val_loader:", len(val_loader))
    return train_loader, val_loader
    
def evaluation(model, dataloader):
    total, correct = 0, 0
    #keeping the network in evaluation mode 
    model.eval()
    for data in dataloader:
        inputs, labels = data
        #moving the inputs and labels to gpu
        #inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total


def train():

    model = CustomNet()
    full_model_path = 'lenet8_version2.pth'
    checkpoint = torch.load(full_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print(model.eval())
    for parameter in model.cnn_model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc_model.parameters():
        parameter.requires_grad = True

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.fc_model.parameters(), lr=0.001)

    train_loader, val_loader = data_loader()

    iter = []
    loss_arr = []
    loss_epoch_arr = []
    max_epochs = 10 #epoch count

    with open('accuracy.json') as f:
        data = json.load(f)
    val_max_acc = data["val_acc"]
    print(val_max_acc)

    for epoch in range(max_epochs):

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            # print(labels)
            # inputs, labels = inputs.to(device), labels.to(device)
            # print()
            #forward pass
            outputs = model(inputs)
            # print('a')
            loss = loss_fn(outputs, labels)

            #backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i+1)%4 ==0:
                iter.append(i)
                loss_arr.append(loss.item())
                val_acc = evaluation(model, val_loader)
                train_acc = evaluation(model, train_loader)
                print('Epoch: %d/%d, Iterations: %d/%d, Loss: %0.4f, Test acc: %0.2f, Train acc: %0.2f' % (epoch+1, max_epochs, i+1, len(train_loader), loss.data, val_acc, train_acc))
                if val_acc>val_max_acc:
                    print('Validation accuracy improved from %0.4f to %0.4f. Saving best model.....' % (val_max_acc,val_acc))
                    val_max_acc = val_acc
                    full_model_path = 'lenet8_full_transfer.pth'
                    torch.save(model, full_model_path)
                else:
                    print('Validation accuracy did not improved from %0.4f. Continuing training.....' % (val_max_acc))
        loss_epoch_arr.append(loss.item())


    
