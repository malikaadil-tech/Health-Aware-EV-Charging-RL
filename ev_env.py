import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EVChargingEnv(gym.Env):

    def __init__(self):
        super(EVChargingEnv, self).__init__()

        # State: [SOC, SOH, price, time_remaining]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )

        # Action: charging/discharging power [-1, 1]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        self.max_steps = 24
        self.alpha = 0.0015  # degradation coefficient

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.soc = 0.5  # start mid-level
        self.soh = 1.0
        self.step_count = 0

        # Time-of-use electricity pricing (sinusoidal)
        base_price = np.sin(np.linspace(0, 3.14, self.max_steps)) * 0.2 + 0.3
        noise = np.random.uniform(-0.05, 0.05, self.max_steps)
        self.price_profile = np.clip(base_price + noise, 0.1, 0.6)

        return self._get_state(), {}

    def _get_state(self):
        index = min(self.step_count, self.max_steps - 1)

        price = self.price_profile[index]
        time_remaining = (self.max_steps - self.step_count) / self.max_steps

        return np.array(
            [self.soc, self.soh, price, time_remaining],
            dtype=np.float32
        )

    def step(self, action):

        action = float(action[0])
        index = min(self.step_count, self.max_steps - 1)
        price = self.price_profile[index]

        # ------------------------------
        # HARD SOC SAFETY CONSTRAINT
        # ------------------------------

        proposed_soc = self.soc + action * 0.05

        # If action would violate SOC bounds, block it
        if proposed_soc < 0 or proposed_soc > 1:
            action = 0.0
            proposed_soc = self.soc

        self.soc = np.clip(proposed_soc, 0, 1)

        # ------------------------------
        # Physics-Inspired Degradation
        # ------------------------------

        degradation = self.alpha * (abs(action) ** 1.5) * (1 + 0.5 * (1 - self.soh))
        self.soh -= degradation
        self.soh = np.clip(self.soh, 0, 1)

        # ------------------------------
        # Economic Objective
        # ------------------------------

        profit = -action * price

        # ------------------------------
        # Multi-Objective Reward
        # ------------------------------

        # Encourage SOC to stay around 50%
        soc_penalty = 3 * abs(self.soc - 0.5)

        # Final reward
        reward = profit - 10 * degradation - soc_penalty

        # ------------------------------

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, False, {}