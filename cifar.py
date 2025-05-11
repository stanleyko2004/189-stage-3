import torch
import torchvision.transforms as transforms
import PIL
from data_loader import load_cifar
from torch.utils.data import Dataset, DataLoader, random_split

class CIFARDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        image = instance['image']
        label = instance['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load data using data_loader
data = load_cifar()

# Create datasets
train_dataset = CIFARDataset(data['train'], transform=train_transform)
test_dataset = CIFARDataset(data['test'], transform=test_transform)

# Split training data into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)
