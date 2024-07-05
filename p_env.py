import numpy as np

import gymnasium as gym
from gymnasium import spaces

class PEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # protocols = [[p1,p2],[F1,F2]] first array contains native operation that generates 2 links, second contains purification schemes, must fit F = 1-lambda*p

    
    
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
        self.lam = -(self.protocols[1][1]-self.protocols[1][0])/(self.protocols[0][1]-self.protocols[0][0])
        self.purification_schemes = 1

        #decide bin size
        self.bin_size = np.floor(np.log((self.F_max-0.25)/(self.threshold-0.25))/self.decay_rate).astype(int)
        self.top_bin = np.floor(np.log((1-0.25)/(self.threshold-0.25))/self.decay_rate).astype(int)
        self.bin_size_min = np.floor(np.log((self.F_min-0.25)/(self.threshold-0.25))/self.decay_rate).astype(int)
        self.f_for_bins = []
        for n in range(self.bin_size_min,self.top_bin+1):
            self.f_for_bins.append((self.threshold-1/4)*np.exp(self.decay_rate*n)+1/4)
        
        print('top bin:',self.top_bin)
        print('top native bin:',self.bin_size)
        print('lowest native bin:',self.bin_size_min)
        print('number of links:',self.num_qubits)

        self.observation_space = spaces.Sequence(spaces.Box(0, self.bin_size - 1, dtype=int))

        self.action_space = spaces.MultiDiscrete([self.purification_schemes+1,self.top_bin])
        self.actions = []
        self.generate_actions()
        self.epl()
        self.combine()
        print(self.actions)


    def generate_actions(self):
        self.actions.append([])
        native_actions = self.actions[0]
        for n in range(self.top_bin+1):
            f = (self.threshold-1/4)*np.exp(self.decay_rate*n)+1/4
            p = (1-f)/self.lam
            if n<=self.bin_size and n>=self.bin_size_min and n>=0:
                native_actions.append(p)
            else:
                native_actions.append(-1)
        return
    
    def epl(self):
        self.actions.append([])
        epl = self.actions[1]
        for n in range(self.top_bin):
            epl.append(-1)
        f = 2/3
        if 1-self.lam*self.p_max>2/3:
            f = 1-self.lam*self.p_max
        if 1-self.lam*self.p_min<2/3:
            f = 1-self.lam*self.p_min
        epl.append(f*f*(1-f)/self.lam)
        return (1-f)/self.lam
        

    #take the maximum probability action of all the purifications
    def combine(self):
        return




    def _get_obs(self):
        return self.memory
        
    def _get_info(self):
        return
        
    def step(self, action):
        # decay step
        self.memory = list(filter(lambda x:x>=0,map(lambda x:x-1,self.memory)))
        p = self.actions[action[0]][action[1]]
        success = np.random.binomial(n=1,p=p)
        if success == 1:
            self.memory.append(action[1])
            if action[0] == 0:               
                self.memory.append(action[1])
            
    # An episode is done iff the agent has reached the target
        terminated = len(self.memory) >= self.num_qubits
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
    
    def run(self, policy):
        obs =[]
        observation, info = self.reset()
        obs.append(observation)
        while(True):
            action = policy[len(obs[-1])]
            observation, reward, terminated, truncated, info = self.step(action)
            obs.append(observation)
            # print(observation)
            if terminated:
                break
        return len(obs)
    
    def close(self):
        return
    


