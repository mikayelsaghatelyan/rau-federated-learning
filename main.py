import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import copy
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 5
NUM_CLIENTS = 10
FRACTION_CLIENTS = 1.0  # fraction of clients to use in each round
NUM_ROUNDS = 50  # federated learning rounds

# constants are mean and standard deviation values for each channel (red, green, blue) of the CIFAR-10 dataset.
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# loading dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Create test data loader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# IID - independent and identically distributed data - независимые и одинаково распределенные данные
# splitting dataset among clients (non-IID) using Dirichlet distribution to simulate realistic non-IID data
# lower alpha means more non-IID (e.g., 0.1 is very skewed, 100.0 is almost uniform)
def create_client_datasets(dataset, num_clients=10, alpha=0.5):
    num_classes = 10
    num_samples = len(dataset)
    labels = np.array([dataset[i][1] for i in range(num_samples)])
    
    # initializing client data indices
    client_data_indices = [[] for _ in range(num_clients)]
    
    # distributing samples according to Dirichlet distribution for each class
    for k in range(num_classes):
        class_indices = np.where(labels == k)[0]
        np.random.shuffle(class_indices)
        
        # sampling from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Distribute indices according to proportions
        proportions = np.array([p*(len(class_indices)//1) for p in proportions])
        proportions = proportions.astype(int)
        proportions[0] += len(class_indices) - proportions.sum()
        
        # assigning indices to clients
        index = 0
        for client_id, prop in enumerate(proportions):
            client_data_indices[client_id].extend(class_indices[index:index+prop].tolist())
            index += prop
    
    # shuffling the indices for each client
    for client_id in range(num_clients):
        np.random.shuffle(client_data_indices[client_id])
    
    # subset datasets for each client
    client_datasets = [Subset(dataset, indices) for indices in client_data_indices]
    
    # distribution statistics
    print("Data distribution among clients:")
    for i, indices in enumerate(client_data_indices):
        client_labels = [labels[j] for j in indices]
        class_counts = [client_labels.count(k) for k in range(num_classes)]
        distribution = [count / len(indices) for count in class_counts]
        print(f"Client {i}: {len(indices)} samples, distribution: {[f'{d:.3f}' for d in distribution]}")
    
    return client_datasets


# trains client model for one epoch
def train_client(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return model, avg_loss, accuracy


# evaluates the model
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy


# weight averaging (FedAvg) - averaging the model weights of client models
def federated_averaging(client_models):
    global_model = CNN().to(device)
    
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    
    global_model.load_state_dict(global_dict)
    
    return global_model


def federated_learning():

    global_model = CNN().to(device)
    
    print("Creating client datasets...")
    client_datasets = create_client_datasets(train_dataset, num_clients=NUM_CLIENTS, alpha=0.5)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    
    round_accuracies = []
    client_accuracies = []

    first_check = True
    
    # learning rounds
    for round_idx in range(NUM_ROUNDS):
        print(f"\n--- Round {round_idx+1}/{NUM_ROUNDS} ---")
        
        # selecting clients for given round
        num_selected = max(1, int(FRACTION_CLIENTS * NUM_CLIENTS))
        selected_clients = np.random.choice(range(NUM_CLIENTS), num_selected, replace=False)
        
        client_models = []
        client_round_accuracies = []
        
        # train each selected client
        for client_idx in selected_clients:
            print(f"Training client {client_idx}...")
            
            # initializing client model with global weights
            client_model = copy.deepcopy(global_model)
            
            # setting up optimizer for client
            optimizer = optim.SGD(client_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
            
            # training for NUM_EPOCHS
            for epoch in range(NUM_EPOCHS):
                client_model, avg_loss, accuracy = train_client(
                    client_model, client_loaders[client_idx], optimizer, epoch
                )
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # evaluating client model on test set
            client_test_loss, client_test_accuracy = evaluate(client_model, test_loader)
            print(f"  Client {client_idx} Test - Loss: {client_test_loss:.4f}, Accuracy: {client_test_accuracy:.2f}%")
            
            client_models.append(client_model)
            client_round_accuracies.append(client_test_accuracy)
        
        # averaging client models to create new global model (FedAvg)
        global_model = federated_averaging(client_models)
        
        # evaluating the global model
        test_loss, test_accuracy = evaluate(global_model, test_loader)
        print(f"Global Model Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
        
        round_accuracies.append(test_accuracy)
        client_accuracies.append(client_round_accuracies)
        
        # accuracy check
        if test_accuracy >= 80.0 and first_check:
            first_check = False
            print(f"\nTarget accuracy of 80% reached at round {round_idx+1}!")
            if round_idx+1 < NUM_ROUNDS:
                continue_training = input("Continue training? (y/n): ")
                if continue_training.lower() != 'y':
                    break
    
    # accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(round_accuracies)+1), round_accuracies, 'b-', marker='o')
    plt.title('Federated Learning Progress - Global Model Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.savefig('fedavg_accuracy.png')
    plt.show()
    
    # client vs global model performance
    plt.figure(figsize=(12, 7))
    round_indices = list(range(1, len(client_accuracies)+1))
    client_avg_acc = [sum(round_acc)/len(round_acc) for round_acc in client_accuracies]
    
    plt.plot(round_indices, round_accuracies, 'b-', marker='o', label='Global Model')
    plt.plot(round_indices, client_avg_acc, 'r--', marker='x', label='Average Client Model')
    
    plt.title('Global vs. Client Models Accuracy')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('global_vs_client_accuracy.png')
    plt.show()
    
    return global_model, round_accuracies


if __name__ == "__main__":
    print("Starting Federated Learning with CIFAR-10...")
    final_model, accuracies = federated_learning()
    
    final_loss, final_accuracy = evaluate(final_model, test_loader)
    print(f"\nFinal Model - Test Accuracy: {final_accuracy:.2f}%")
    
    torch.save(final_model.state_dict(), 'federated_cifar10_model.pth')
    print("Model saved to 'federated_cifar10_model.pth'")