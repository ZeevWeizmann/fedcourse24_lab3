import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

DEVICE = torch.device("cpu")

# -------------------------------
# Простая модель для CIFAR10
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# -------------------------------
# Загрузка CIFAR10
# -------------------------------
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# -------------------------------
# Клиент Federated Distillation
# -------------------------------
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        logits_list = []
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                logits_list.append(outputs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_logits = np.mean(np.vstack(logits_list))  # <-- усредняем до одного скаляра

        print(f"[Client] Accuracy: {accuracy:.4f}, Avg logit: {avg_logits:.4f}")

        # Возвращаем только скаляры
        return float(0.0), len(testloader.dataset), {
            "accuracy": float(accuracy),
            "avg_logits": float(avg_logits),
        }
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=CifarClient())
