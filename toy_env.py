import numpy as np

import gymnasium as gym
from gymnasium import spaces

class ToyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # protocols = [[p1,p2,...],[F1,F2,...]]

    def __init__(self, protocols, num_qubits, threshold, decay_rate):
        self.num_qubits = num_qubits
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.protocols = protocols
        self.memory = []

        #decide bin size
        F_max = np.max(protocols[1])
        F_min = np.min(protocols[1])
        self.bin_size = np.floor(np.log((F_max-0.25)/(self.threshold-0.25))/self.decay_rate).astype(int)
        self.bin_size_min = np.floor(np.log((F_min-0.25)/(self.threshold-0.25))/self.decay_rate).astype(int)
        self.bins = []
        for f in protocols[1]:
            bin = np.floor(np.log((f-0.25)/(self.threshold-0.25))/self.decay_rate)
            self.bins.append(bin)

        
        self.observation_space = spaces.Sequence(spaces.Box(0, self.bin_size - 1, dtype=int))

        self.action_space = spaces.Discrete(len(protocols[0]))
        print('the total bin size is:')
        print(self.bin_size)
        print('the bins are:')
        print(self.bins)

    def _get_obs(self):
        return self.memory
        
    def _get_info(self):
        return
        
    def step(self, action):
    # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.memory = list(filter(lambda x:x>=0,map(lambda x:x-1,self.memory)))
        success = np.random.binomial(n=1,p=self.protocols[0][action])
        if success == 1:
            bin_pos = self.bins[action]
            self.memory.append(bin_pos)
            
    # An episode is done iff the agent has reached the target
        terminated = len(self.memory) == self.num_qubits
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        truncated = False

        return observation, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.memory = []

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def close(self):
        return
    


