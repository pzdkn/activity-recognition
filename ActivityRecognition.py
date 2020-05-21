from SimpleClassifier import SimpleClassifier
from AcitivityData import ActivityDataset
from AcitivityData import train_dev_test_loader
from AcitivityData import homogenous_trans, pseudo_relative_trans, pseudo_toy_trans
import torch.nn as nn
from torch import optim
import torch
from sklearn import metrics
import matplotlib.pyplot as plt

def train(trainloader, devloader, model, epochs=10, lr=0.001, device="cpu"):
    model.to(device)
    # Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    steps = 0
    running_loss = 0
    print_every = 1000
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for i, traindata in enumerate(trainloader):
            inputs, labels = traindata['trajectory'], traindata['label']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model.forward(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for devdata in devloader:
                        inputs, labels = devdata['trajectory'], devdata['label']
                        inputs, labels = inputs.to(device), labels.to(device)
                        logits = model.forward(inputs)
                        batch_loss = criterion(logits, labels)
                        test_loss += batch_loss.item()
                        _, top_class = torch.max(logits, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(devloader))
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Dev loss: {test_loss / len(devloader):.3f}.. "
                      f"Dev accuracy: {accuracy / len(devloader):.3f}")
                running_loss = 0
                model.train()
    return train_losses, test_losses


def test(tesloader, model, device="cpu"):
    accuracy = 0
    preds = torch.tensor([], dtype=torch.long)
    labels = torch.tensor([], dtype=torch.long)
    with torch.no_grad():
        for data in tesloader:
            inputs, label = data['trajectory'], data['label']
            inputs, label = inputs.to(device), label.to(device)
            logits = model.forward(inputs)
            _, top_class = torch.max(logits, dim=1)
            equals = top_class == label.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            preds = torch.cat((preds, top_class), dim=0)
            labels = torch.cat((labels, label), dim=0)
        print(f"Test accuracy: {accuracy / len(testloader):.3f}")
    return preds, labels

def compute_metrics(preds, labels, label_dict):
    metric_dict={}
    f1 = metrics.f1_score(labels, preds, average=None)
    precision = metrics.precision_score(labels, preds, average=None)
    recall = metrics.recall_score(labels, preds, average=None)
    for key, value in label_dict.items():
        metric_dict[value] = {"precision": precision[key], "recall": recall[key], "f1": f1[key]}
    return metric_dict

if __name__ == '__main__':
    act_data = ActivityDataset('./Data/27_04', window_length=5)
    label_dict = act_data.labels
    trainloader, devloader, testloader = train_dev_test_loader(act_data)
    model = SimpleClassifier(len(label_dict))
    model.double()
    train_losses, test_losses = train(trainloader, devloader, model)
    plt.plot(train_losses, label="Traing Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend(loc='lower right')
    plt.show()
    preds, labels = test(testloader, model)
    metric_dict = compute_metrics(preds, labels, label_dict)
    print(metric_dict)
