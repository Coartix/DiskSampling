import numpy as np
import cv2  # Import OpenCV
import matplotlib.pyplot as plt
import random

import numpy as np
import random


class DiskEnv:
    def __init__(self, size=128, num_actions=128*128):
        self.size = size
        self.radius = None
        self.target_coverage = None
        self.total_area = size * size
        self.circle_area = None
        self.circle_centers = []
        self.num_disks = None
        self.grid_size = int(np.sqrt(num_actions))
        self.reset()

    def reset(self):
        self.circle_centers.clear()
        # self.radius = random.uniform(3, 6)
        # Constant for now
        self.radius = 4
        # self.target_coverage = random.uniform(0.55, 0.8)
        # Constant for now
        self.target_coverage = 0.45
        self.circle_area = np.pi * self.radius**2
        self.num_disks = int(self.target_coverage * self.total_area / self.circle_area)
        return self.get_state()
    
    def step(self, action):
        x, y = action
        reward = 0
        done = False

        if self._can_place_circle(x, y):
            self.circle_centers.append((x, y))
            reward = len(self.circle_centers)**2

        if len(self.circle_centers) >= self.num_disks:
            done = True
            reward += 1e5

        return self.get_state(), reward, done

    def _can_place_circle(self, x, y):
        min_squared_distance = (2 * self.radius)**2

        for center in self.circle_centers:
            dx = min(abs(center[0] - x), self.size - abs(center[0] - x))
            dy = min(abs(center[1] - y), self.size - abs(center[1] - y))
            squared_distance = dx**2 + dy**2

            if squared_distance < min_squared_distance:
                return False
        return True

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for (x, y) in self.circle_centers:
            state[x % self.grid_size, y % self.grid_size, 0] = 1.0
        return state
    
    def render(self, title="Board Env"):
        upscale_factor = 100
        upscaled_size = self.size * upscale_factor

        state = np.zeros((upscaled_size, upscaled_size), dtype=np.uint8)

        for (x, y) in self.circle_centers:
            upscaled_x = int(x * upscale_factor)
            upscaled_y = int(y * upscale_factor)
            upscaled_radius = int(self.radius * upscale_factor)
            cv2.circle(state, (upscaled_x, upscaled_y), upscaled_radius, 255, -1)

            # Draw the circle at the wrapped locations if it crosses the edges
            if upscaled_x - upscaled_radius < 0:
                cv2.circle(state, (upscaled_x + upscaled_size, upscaled_y), upscaled_radius, 255, -1)
            if upscaled_x + upscaled_radius > upscaled_size:
                cv2.circle(state, (upscaled_x - upscaled_size, upscaled_y), upscaled_radius, 255, -1)
            if upscaled_y - upscaled_radius < 0:
                cv2.circle(state, (upscaled_x, upscaled_y + upscaled_size), upscaled_radius, 255, -1)
            if upscaled_y + upscaled_radius > upscaled_size:
                cv2.circle(state, (upscaled_x, upscaled_y - upscaled_size), upscaled_radius, 255, -1)

            # Handle corners
            if upscaled_x - upscaled_radius < 0 and upscaled_y - upscaled_radius < 0:
                cv2.circle(state, (upscaled_x + upscaled_size, upscaled_y + upscaled_size), upscaled_radius, 255, -1)
            if upscaled_x + upscaled_radius > upscaled_size and upscaled_y - upscaled_radius < 0:
                cv2.circle(state, (upscaled_x - upscaled_size, upscaled_y + upscaled_size), upscaled_radius, 255, -1)
            if upscaled_x - upscaled_radius < 0 and upscaled_y + upscaled_radius > upscaled_size:
                cv2.circle(state, (upscaled_x + upscaled_size, upscaled_y - upscaled_size), upscaled_radius, 255, -1)
            if upscaled_x + upscaled_radius > upscaled_size and upscaled_y + upscaled_radius > upscaled_size:
                cv2.circle(state, (upscaled_x - upscaled_size, upscaled_y - upscaled_size), upscaled_radius, 255, -1)

        resized_state = cv2.resize(state, (self.size, self.size), interpolation=cv2.INTER_AREA)

        plt.imshow(resized_state, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
