from SimpleClassifier import SimpleClassifier
from ActivityData import ActivityDataset
import torch.nn as nn
from torch import optim
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import random


def train(trainloader, devloader, model, epochs=3, lr=0.001, savepath="./act_rec.pth", device="cpu"):
    model.to(device)
    # Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_loss = 0
    print_every = 1000
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for i, traindata in enumerate(trainloader):
            inputs, labels = traindata['trajectory'], traindata['activity_label']
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
                        inputs, labels = devdata['trajectory'], devdata['activity_label']
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


def evaluate_action(testloader, model, obj2idx, labels2idx, device='cpu'):
    """
    solely evaluates single action
    """
    accuracy = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            object_index = data['object_label']
            trajectory = data['trajectory']
            traj = trajectory[:, :, [0, object_index], :]
            activity_index = data['activity_label']
            traj, activity_index = traj.to(device), activity_index.to(device)
            logits = model.forward(traj)
            _, top_class = torch.max(logits, dim=1)
            equals = top_class == activity_index.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            total += 1
            for obj_idx in object2idx.values():
                if obj_idx in (0, object_index):
                    continue
                na_index = torch.tensor([label2idx['no action']], dtype=torch.int64)
                na_traj = trajectory[:, :, [0, obj_idx], :]
                logits = model.forward(na_traj)
                _, top_class = torch.max(logits, dim=1)
                equals = top_class == na_index.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                total += 1
        print(f"Test accuracy: {accuracy / total :.3f}")


def evaluate_action_object(testloader, model, obj2idx, labels2idx, device="cpu"):
    """
    iterates over all object trajectories, predicts the action, object pair that
    maximizes the logits
    """
    accuracy = 0
    with torch.no_grad():
        for data in testloader:
            object_index = data['object_label']
            trajectory = data['trajectory']
            activity_index = data['activity_label']
            overall_max = -1e3
            for obj_label, obj_index in obj2idx.items():
                if obj_index == 0:
                    continue
                logits = model.forward(trajectory[:, :, [0, object_index], :])
                max_value, max_index = torch.max(logits, dim=1)
                if max_value > overall_max:
                    object_pred = object_index
                    activity_pred = max_index
                    overall_max = max_value
            if object_pred == object_index and activity_pred == activity_index:
                accuracy += 1
        print(f"Action Object Test Accuracy: {accuracy / len(testloader):.3f}")


def compute_metrics(preds, labels, label_dict):
    metric_dict = {}
    f1 = metrics.f1_score(labels, preds, average=None)
    precision = metrics.precision_score(labels, preds, average=None)
    recall = metrics.recall_score(labels, preds, average=None)
    for key, value in label_dict.items():
        metric_dict[key] = {"precision": precision[value], "recall": recall[value], "f1": f1[value]}
    return metric_dict


def print_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Traing Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Losses")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(42)
    print(random.random())
    act_data = ActivityDataset('./Data/27_04', window_length=5)
    label2idx = act_data.labels2idx
    object2idx = act_data.object2idx
    # data loader get splitted and shuffled data
    trainloader, devloader, testloader = act_data.train_dev_test_loader()
    # initialize classifier with specified output dimension
    model = SimpleClassifier(len(label2idx))
    model.double()
    train_losses, test_losses = train(trainloader, devloader, model)
    print_losses(train_losses, test_losses)
    evaluate_action(testloader, model, object2idx, label2idx)
    evaluate_action_object(testloader, model, object2idx, label2idx)
