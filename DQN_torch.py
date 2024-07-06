import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Any

class DQN(nn.Module):
    def __init__(self, grid_size: int, metadata_size: int, num_actions: int):
        super(DQN, self).__init__()
        self.grid_size = grid_size
        self.metadata_size = metadata_size
        self.num_actions = num_actions
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size of the flattened conv output
        conv_out_size = self._get_conv_out_size(grid_size)

        # Fully connected layer for combining conv output and metadata
        self.fc_combined = nn.Linear(conv_out_size + metadata_size, 2048)

        # Policy network with additional fully connected layer and deconvolution layers
        self.policy_fc = nn.Linear(2048, 128*4*4)

        self.policy_net = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out_size(self, grid_size: int) -> int:
        o = torch.zeros(1, 1, grid_size, grid_size)
        o = self.conv1(o)
        o = self.bn1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.conv3(o)
        o = self.bn3(o)
        return int(np.prod(o.size()))

    def forward(self, state: torch.Tensor, metadata: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the tensor
        x = x.reshape(batch_size, -1)

        # Combine with metadata
        combined = torch.cat((x, metadata), dim=1)
        combined = F.relu(self.fc_combined(combined))

        # PolicyNet
        policy_fc_output = F.relu(self.policy_fc(combined)).view(batch_size, 128, 4, 4)
        policy_output = self.policy_net(policy_fc_output)
        policy_output = policy_output.view(batch_size, self.num_actions)

        # ValueNet
        value_output = self.value_net(combined)

        return policy_output, value_output