import numpy as np
import torch
from datasets import load_data
from torch_model import resnet18
import tensorflow as tf

# torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
num_classes = 10

def main():
    train_loader, test_loader = load_data(batch_size)

    # torch model test
    pt = '.\\model\\resnet18.pt'
    model = resnet18(num_classes).to(device)
    weights = torch.load(pt, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)  # batch size
            correct += (pred == labels).sum().item()

        print(f'torch test accuracy: {100 * (correct / total): .2f}%')

    # tf model test
    pb = '.\\model\\resnet18_save_model'
    tf_model = tf.keras.models.load_model(pb)
    total, correct = 0, 0
    for images, labels in test_loader:
        images = images.permute(0, 2, 3, 1).numpy()
        total += labels.size(0)
        labels = labels.numpy()
        pred = tf_model.predict(images)
        pred = np.argmax(pred, axis=1)
        correct += np.sum(pred == labels)
    # print(total, correct)
    acc = correct / total
    print(f'tf test accuracy: {acc * 100:.2f}%')


if __name__ == '__main__':
    main()