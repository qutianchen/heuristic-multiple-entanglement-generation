import numpy as np

import gymnasium as gym
from gymnasium import spaces

class IntpAltEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # protocols = [[p1,p2],[F1,F2]] assuming linear F-p relation

    
    
    def __init__(self, protocols, num_qubits, threshold, decay_rate):
        self.num_qubits = num_qubits
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.protocols = protocols
        self.memory = []
        self.F_min = np.amin(protocols[1])
        self.F_max = np.amax(protocols[1])
        self.p_min = np.amin(protocols[0])
        self.p_max = np.amax(protocols[0])

        #decide bin size
        self.bin_size = np.floor(np.log((self.F_max-0.25)/(self.threshold-0.25))/self.decay_rate)
        self.bin_size_min = np.floor(np.log((self.F_min-0.25)/(self.threshold-0.25))/self.decay_rate)
        self.observation_space = spaces.Sequence(spaces.Box(0, self.bin_size - 1, dtype=int))

        self.action_space = spaces.Box(0,1,dtype=np.float32)
        print('the upper bin size is:')
        print(self.bin_size)
        print('the lower bin size')
        print(self.bin_size_min)


    def _get_obs(self):
        return self.memory
        
    def _get_info(self):
        return
        
    def step(self, q):
    # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.memory = list(filter(lambda x:x>=0,map(lambda x:x-1,self.memory)))
        choice = np.random.binomial(n=1,p=q)
        success, f = 0, 0
        if choice == 0:
            success = np.random.binomial(n=1,p=np.min(self.protocols[0]))
        else:
            success = np.random.binomial(n=1,p=np.max(self.protocols[0]))
        if success == 1:
            if choice == 0:
                f = self.F_max
            else:
                f = self.F_min
            bin_pos = np.floor(np.log((f-0.25)/(self.threshold-0.25))/self.decay_rate)
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
    


