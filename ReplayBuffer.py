from collections import deque
import random
import numpy as np
import torch
from typing import List, Tuple, Any

'''class ReplayBuffer:
    def __init__(self, device: torch.device, size: int = 10000, grid_size: int = 128):
        """
        Initialize the ReplayBuffer.

        Parameters:
        - device: The device (CPU or CUDA) where tensors should be sent.
        - size: The maximum size of the buffer.
        - grid_size: The size of the grid (width and height).
        """
        self._maxsize = size
        self._storage = deque(maxlen=self._maxsize)
        self.device = device
        self.grid_size = grid_size

    def __len__(self) -> int:
        """
        Return the current size of the buffer.

        Returns:
        - The number of elements in the buffer.
        """
        return len(self._storage)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, radius: int) -> None:
        """
        Add a new experience to the buffer.

        Parameters:
        - state: The current state grid.
        - action: The action taken in the state.
        - reward: The reward received after taking the action.
        - next_state: The state grid after taking the action.
        - done: Boolean indicating if the episode ended after taking the action.
        - radius: The radius used for the current state.
        """
        state = state.squeeze(0).transpose(1, 2, 0)  # Convert from (1, 1, grid_size, grid_size) to (grid_size, grid_size, 1)
        next_state = next_state.squeeze(0).transpose(1, 2, 0)
        state_coords = self.grid_to_coordinates(state)
        next_state_coords = self.grid_to_coordinates(next_state)
        data = (state_coords, action, reward, next_state_coords, done, radius)
        self._storage.append(data)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.

        Parameters:
        - batch_size: The size of the batch to sample.

        Returns:
        - A tuple containing tensors of states, actions, rewards, next_states, dones, and radii.
        """
        batch = random.sample(self._storage, batch_size)
        batch = np.array(batch, dtype=object)

        states = np.array([self.coordinates_to_grid(obs[0]) for obs in batch])
        actions = np.array([obs[1] for obs in batch])
        rewards = np.array([obs[2] for obs in batch])
        next_states = np.array([self.coordinates_to_grid(obs[3]) for obs in batch])
        dones = np.array([obs[4] for obs in batch])
        radii = np.array([obs[5] for obs in batch])

        states = torch.tensor(states).float().to(self.device).permute(0, 3, 1, 2)  # Convert to (batch_size, 1, grid_size, grid_size)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device).permute(0, 3, 1, 2)  # Convert to (batch_size, 1, grid_size, grid_size)
        dones = torch.tensor(dones).float().to(self.device)
        radii = torch.tensor(radii).float().to(self.device)

        return states, actions, rewards, next_states, dones, radii

    def grid_to_coordinates(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert a grid state back to a list of coordinates.

        Parameters:
        - grid: A numpy array representing the grid state.

        Returns:
        - A list of tuples representing the coordinates of the disks.
        """
        coords = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j, 0] == 1.0:
                    coords.append((i, j))
        return coords

    def coordinates_to_grid(self, coordinates: List[Tuple[int, int]]) -> np.ndarray:
        """
        Convert the coordinates of circles back to a state representation (grid_size x grid_size x 1).

        Parameters:
        - coordinates: List of tuples representing the centers of the disks.

        Returns:
        - A numpy array representing the state with shape (grid_size, grid_size, 1).
        """
        state = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for x, y in coordinates:
            state[x, y, 0] = 1.0
        return state'''

class ReplayBuffer:
    def __init__(self, device, initial_size=10000, max_size=50000, grid_size=128):
        self._storage = []
        self._backup = []
        self.device = device
        self.grid_size = grid_size
        self.max_size_backup = max_size
        self.size = initial_size

    def __len__(self):
        return len(self._storage)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, radius: int) -> None:
        """
        Add a new experience to the buffer.

        Parameters:
        - state: The current state grid.
        - action: The action taken in the state.
        - reward: The reward received after taking the action.
        - next_state: The state grid after taking the action.
        - done: Boolean indicating if the episode ended after taking the action.
        - radius: The radius used for the current state.
        """
        state = state.squeeze(0).transpose(1, 2, 0)  # Convert from (1, 1, grid_size, grid_size) to (grid_size, grid_size, 1)
        next_state = next_state.squeeze(0).transpose(1, 2, 0)
        state_coords = self.grid_to_coordinates(state)
        next_state_coords = self.grid_to_coordinates(next_state)
        data = (state_coords, action, reward, next_state_coords, done, radius)

        if len(self._storage) > self.size:
            self._backup.append(data)
            if len(self._backup) > self.max_size_backup:
                # Choose randomly self.size elements from self._backup and replace self._storage with them
                self._storage = random.sample(self._backup, self.size)
                self._backup = []
        else:
            self._storage.append(data)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.

        Parameters:
        - batch_size: The size of the batch to sample.

        Returns:
        - A tuple containing tensors of states, actions, rewards, next_states, dones, and radii.
        """
        batch = random.sample(self._storage, batch_size)
        batch = np.array(batch, dtype=object)

        states = np.array([self.coordinates_to_grid(obs[0]) for obs in batch])
        actions = np.array([obs[1] for obs in batch])
        rewards = np.array([obs[2] for obs in batch])
        next_states = np.array([self.coordinates_to_grid(obs[3]) for obs in batch])
        dones = np.array([obs[4] for obs in batch])
        radii = np.array([obs[5] for obs in batch])

        states = torch.tensor(states).float().to(self.device).permute(0, 3, 1, 2)  # Convert to (batch_size, 1, grid_size, grid_size)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device).permute(0, 3, 1, 2)  # Convert to (batch_size, 1, grid_size, grid_size)
        dones = torch.tensor(dones).float().to(self.device)
        radii = torch.tensor(radii).float().to(self.device)
        return states, actions, rewards, next_states, dones, radii

    def grid_to_coordinates(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert a grid state back to a list of coordinates.

        Parameters:
        - grid: A numpy array representing the grid state.

        Returns:
        - A list of tuples representing the coordinates of the disks.
        """
        coords = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j, 0] == 1.0:
                    coords.append((i, j))
        return coords

    def coordinates_to_grid(self, coordinates: List[Tuple[int, int]]) -> np.ndarray:
        """
        Convert the coordinates of circles back to a state representation (grid_size x grid_size x 1).

        Parameters:
        - coordinates: List of tuples representing the centers of the disks.

        Returns:
        - A numpy array representing the state with shape (grid_size, grid_size, 1).
        """
        state = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for x, y in coordinates:
            state[x, y, 0] = 1.0
        return state
