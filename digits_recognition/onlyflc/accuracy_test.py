import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns


num_epochs = 1

possible_learning_rates = [0.005, 0.01, 0.25, 0.05, 0.075]
possible_batch_sizes = [4,8,16,32,64]


for i, bs in enumerate(possible_batch_sizes):
    for j, lr in enumerate(possible_learning_rates):
        learning_rate = lr
        batch_size = bs


        #loading and normalizing the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n Using device: {device}\n")

        #defining the model
        '''class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(28*28, 512)
                self.fc2 = nn.Linear(512,256)
                self.fc3 = nn.Linear(256, 10)

            def forward(self, x):
                x = x.view(-1, 28*28)  # Flatten the image
                x = F.relu(self.fc1(x))  # ReLU activation
                x = F.relu(self.fc2(x))
                x = self.fc3(x)  # Output layer
                return x


        model = Net().to(device)'''

        model = nn.Sequential (
            nn.Flatten(),
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )


        #defining loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)


        def accuracy(outputs, labels):
            _, preds = torch.max(outputs, 1)
            return torch.sum(preds == labels).item() / len(labels)

        def train(model, device, train_loader, criterion, optimizer, epoch):
            
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_acc += accuracy(outputs, labels)
                if (i + 1) % 200 == 0:
                    #print(f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
                    running_loss = 0.0
                    running_acc = 0.0
            #print(f'ok {epoch}')        

        def test(model, device, test_loader, criterion):

            model.eval()
            test_loss = 0.0
            test_acc = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:

                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_acc += accuracy(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            conf_matrix = confusion_matrix(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')

            print(f'\nTEST {5*i+j+1}: Learning rate = {learning_rate}, Batch size = {batch_size}\n')
            print(f'\nLoss: {test_loss / len(test_loader):.4f}')
            print(f'Accuracy: {test_acc / len(test_loader):.4f}\n')
            print("Confusion Matrix: \n", conf_matrix, '\n')
            #print("Classification Report:\n")
            #print(classification_report(all_labels, all_preds, digits=4))
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}\n\n")

            '''# Visualize confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
            plt.xlabel("Pred")
            plt.ylabel("Label")
            plt.title("Confusion Matrix")
            plt.savefig("confmat.png")'''


        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, criterion, optimizer, epoch)

        test(model, device, test_loader, criterion)


        '''# Visualize sample images with predictions
        samples, labels = next(iter(test_loader))
        samples = samples.to(device)
        outputs = model(samples)
        _, preds = torch.max(outputs, 1)

        n_lines = 8
        n_columns = 8

        num_examples = n_columns*n_lines
        random_indices = random.sample(range(samples.size(0)), num_examples)

        samples = samples.cpu().numpy()
        fig, axes = plt.subplots(n_lines, n_columns, figsize=(10, 10))
        for i, ax in enumerate(axes.ravel()):
            index = random_indices[i]
            ax.imshow(samples[index].squeeze(), cmap='gray')
            ax.set_title(f'L: {labels[index]}, P: {preds[index]}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("samples.png")
'''
