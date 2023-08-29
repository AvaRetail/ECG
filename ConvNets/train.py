import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# from torch.nn import functional as f
# from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import math
import numpy as np

def prepare_dataset():
    _transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    train_dataset = DataLoader(ImageFolder(r"C:\Users\ATI-G2\Documents\python\ECG\data\dataRoot\train",transform=_transforms), batch_size=8)
    val_dataset = DataLoader(ImageFolder(r"C:\Users\ATI-G2\Documents\python\ECG\data\dataRoot\val", transform=_transforms), batch_size=4)

    return train_dataset, val_dataset

def model_definition():
    model = resnet50(ResNet50_Weights)

    n_classes = 7
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    
    return model.train()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = prepare_dataset()

    model = model_definition().to(device)

    optim = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    n_epochs = 2

    # iter = 0
    val_loss = []
    train_loss = []

    for i in range(n_epochs):
        for images, labels in tqdm(train_dataset, f'epoch-{i+1}/{n_epochs}'):
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            probs = model(images)
            loss = criterion(probs, labels)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())

        print(f"training loss for the epoch-{i+1}/{n_epochs}: {sum(train_loss)/len(train_loss)}")
        train_loss.clear()
    
        with torch.no_grad():

            for images, labels in tqdm(val_dataset, "validating"):
                images, labels = images.to(device), labels.to(device)
                probs = model(images)
                loss = criterion(probs, labels)
                val_loss.append(loss.item())

            print(f"val loss for the epoch-{i+1}/{n_epochs}: {sum(val_loss)/len(val_loss)}")
            val_loss.clear()


if __name__=="__main__":
    main()








    