import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# from torch.nn import functional as f
# from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

def prepare_dataset():
    _transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = DataLoader(ImageFolder(r"C:\Users\ATI-G2\Documents\python\ECG\data\dataRoot\train",transform=_transforms), batch_size=16)
    val_dataset = DataLoader(ImageFolder(r"C:\Users\ATI-G2\Documents\python\ECG\data\dataRoot\val", transform=_transforms), batch_size=4)

    return train_dataset, val_dataset

def model_definition():
    model = resnet50(ResNet50_Weights)
    return model.train()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = prepare_dataset()

    model = model_definition().to(device)

    optim = Adam(model.get_parameter())
    criterion = nn.CrossEntropyLoss()

    n_epochs = 2

    for i in range(n_epochs):
        for images, labels in train_dataset:
            images, labels = images.to(device), labels.to(device)
            probs = model(images)
            optim.zero_grad()
            loss = criterion(probs, labels)
            loss.backwards()

if __name__=="__main__":
    main()








    