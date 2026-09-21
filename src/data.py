from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root = "./data",
        train = True,  
        download = True,
        transform = transform,
    )

    test_dataset = datasets.CIFAR10(
        root = "./data",
        train = False,
        download = True,
        transform = transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle = False,
        num_workers = 2,  
    )

    return train_loader, test_loader