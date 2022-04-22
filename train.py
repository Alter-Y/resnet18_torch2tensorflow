import torch
import torch.nn as nn
from datasets import load_data
from torch_model import resnet18

# torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
num_classes = 10
batch_size = 32
learning_rate = 0.001

def train():
    # load model and dataset
    model = resnet18(num_classes).to(device)
    train_loader, test_loader = load_data(batch_size)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader)
    # train
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # data need in gpu, if model is in gpu
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # bp and optimizer(gradient to zero, error backprop, update para)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.3f}')

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0) # batch size
                correct += (pred == labels).sum().item()

            print(f'Test accuracy: {100 * (correct/total)}%')

    torch.save(model.state_dict(), 'model/resnet18.pt')


if __name__ == '__main__':
    train()