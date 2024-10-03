# 1. Importer les bibliothèques nécessaires
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 2. Définir les transformations pour les images MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertir les images en tenseurs PyTorch
    transforms.Normalize((0,), (1,))  # Normaliser les images
])

# 3. Charger les données MNIST
download = False
train_set = datasets.MNIST(root='./data/', train=True, download=download, transform=transform)
test_set = datasets.MNIST(root='./data/', train=False, download=download, transform=transform)

batch_size = 1000
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# 4. Définir le modèle de réseau neuronal
class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Adapter la taille selon la sortie des convolutions
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64 * 5 * 5)  # Redimensionner pour la couche entièrement connectée
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 5. Initialiser le modèle, la fonction de perte et l'optimiseur
model = MNIST_Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 6. Fonction pour entraîner le modèle
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Réinitialiser les gradients
        output = model(data)  # Faire une prédiction
        loss = criterion(output, target)  # Calculer la perte
        loss.backward()  # Rétropropagation
        optimizer.step()  # Mise à jour des paramètres

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 7. Fonction pour tester le modèle
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Ajouter la perte sur l'ensemble des données de test
            pred = output.argmax(dim=1, keepdim=True)  # Trouver la classe prédite
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    return accuracy

# 8. Configuration pour utiliser le GPU si disponible
device = torch.device("mps")
model.to(device)

from time import time

t0 = time()

# 9. Entraîner le modèle pendant 5 époques
num_epochs = 15
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    print(time()-t0)

# 10. Sauvegarder le modèle après l'entraînement (optionnel)
torch.save(model.state_dict(), "./models/mnist_cnn.pth")