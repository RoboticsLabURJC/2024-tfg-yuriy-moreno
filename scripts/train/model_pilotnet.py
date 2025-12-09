# model_pilotnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PilotNetVW(nn.Module):
    def __init__(self, input_shape=(3, 66, 200)):
        super().__init__()

        # Bloque convolucional tipo PilotNet
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Capa de flatten como módulo para poder medir la salida
        self.flatten = nn.Flatten()

        # Calculamos automáticamente el tamaño de la salida convolucional
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)   # (1, 3, 66, 200)
            x = self._forward_conv(dummy)
            n_flat = x.shape[1]

        # Bloque fully-connected
        self.fc1 = nn.Linear(n_flat, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc_out = nn.Linear(10, 2)  # [v, w]

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc_out(x)
        return out  # (batch_size, 2) -> [v, w]
