import torch
from model import create_model
from train import train_model
from data_loading import CustomImageDataset
from torch.utils.data import DataLoader

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    image_dir = "/Users/mr.nerdhuge/Downloads/sayzek_project/satelite/images"

    train_dataset = CustomImageDataset(image_dir)
    val_dataset = CustomImageDataset(image_dir)

    print(f"Number of images in dataset: {len(train_dataset)}")
    image, target = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Target: {target}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    num_classes = 2
    model = create_model(num_classes)

    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

    torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == '__main__':
    main()
