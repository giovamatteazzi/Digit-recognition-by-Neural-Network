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
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import io
import time


num_epochs = 5

batch_size = 64
learning_rate = 0.0175

transform = transforms.Compose([
    transforms.Resize((32,32)),    #lenet5 only accepts 32x32 inputs
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Using device: {device}\n")

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lay2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(400,120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = LeNet5().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

def train(model, device, train_loader, criterion, optimizer, epoch):
    
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    start_time = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if (i + 1) % 200 == 0:
            #print(f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
            running_loss = 0.0
            running_acc = 0.0
        epoch_time = time.time() - start_time
    print(f"Epoch {epoch} finished in {epoch_time:.3f} seconds")        
    return epoch_time


def test(model, device, test_loader, criterion):

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()

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
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    '''
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    '''
    print(f"\nTest inference finished in {inference_time:.3f} seconds")
    print(f'\nLoss: {test_loss / len(test_loader):.4f}')
    print(f'Accuracy: {test_acc / len(test_loader):.4f}\n')
    '''
    print("Confusion Matrix: \n", conf_matrix, '\n')
    #print("Classification Report:\n")
    #print(classification_report(all_labels, all_preds, digits=4))
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n\n")

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Pred")
    plt.ylabel("Label")
    plt.title("Confusion Matrix")
    plt.savefig("confmat.png")
'''
epoch_times = []

total_start = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_duration = train(model, device, train_loader, criterion, optimizer, epoch)
    epoch_times.append(epoch_duration)

test(model, device, test_loader, criterion)

total_time = time.time() - total_start

print(f"Total time: {total_time:.3f} seconds")
print(f"Average time per epoch: {sum(epoch_times) / len(epoch_times):.3f} seconds\n")

'''# visualize sample images with predictions
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

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Scrivi una cifra (0-9)")
        
        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text='Pulisci', command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.predict_button = tk.Button(self.button_frame, text='Predici', command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT)

        self.label = tk.Label(self.master, text="Scrivi una cifra...")
        self.label.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 6), (event.y - 6)
        x2, y2 = (event.x + 6), (event.y + 6)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Scrivi una cifra...")

    def preprocess_image(self):
        # resizes to 32x32
        image_resized = self.image.resize((32, 32), resample=Image.Resampling.LANCZOS)
        image_inverted = ImageOps.invert(image_resized)

        # normalizes and converts into numpy array
        image_array = np.asarray(image_inverted) / 255.0

        # (1, 1, 28, 28) -> batch x channel x height x width
        input_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return input_tensor

    def predict_digit(self):
        input_tensor = self.preprocess_image()

        # inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item() * 100  # in percentuale

        probs = probabilities.squeeze().cpu().numpy()
        probs_str = " | ".join([f"{i}: {p*100:.1f}%" for i, p in enumerate(probs)])
        self.label.config(text=f"Predizione: {prediction} ({confidence:.1f}%)\n{probs_str}")

# launches the program
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()'''