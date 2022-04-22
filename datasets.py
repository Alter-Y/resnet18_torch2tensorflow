from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_data(batch_size=32):
    # data augmentation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    cifar_train = datasets.CIFAR10('cifar', True, transform=transform, download=True)
    cifar_test = datasets.CIFAR10('cifar', False, transform=transform)

    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    load_data()