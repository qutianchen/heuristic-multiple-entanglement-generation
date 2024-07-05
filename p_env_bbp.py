import numpy as np

import gymnasium as gym
from gymnasium import spaces

class PEnvBBP(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # protocols = [[p1,p2],[F1,F2]] first array contains native operation that generates 2 links, second contains purification schemes, must fit F = 1-lambda*p

    
    
    def __init__(self, protocols, num_qubits, threshold, decay_rate, p_threshold = -1):
        self.num_qubits = num_qubits
        self.threshold = threshold
        self.decay_rate = decay_rate
        self.protocols = protocols
        self.p_threshold_bin = 0 if p_threshold==-1 else np.floor(np.log((p_threshold-0.25)/(self.threshold-0.25))/self.decay_rate).astype(int)
        self.memory = []
        self.F_min = np.amin(protocols[1])
        self.F_max = np.amax(protocols[1])
        self.p_min = np.amin(protocols[0])
        self.p_max = np.amax(protocols[0])
        self.lam = -(self.protocols[1][1]-self.protocols[1][0])/(self.protocols[0][1]-self.protocols[0][0])

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

        self.observation_space = spaces.Sequence(spaces.Box(0, self.bin_size - 1, dtype=int))

        self.action_space = spaces.Discrete(self.top_bin)
        self.generate_actions()
        print(self.actions)


    def generate_actions(self):
        self.actions=[]
        for n in range(self.top_bin+1):
            f = (self.threshold-1/4)*np.exp(self.decay_rate*n)+1/4
            p = (1-f)/self.lam
            if n<=self.bin_size and n>=self.bin_size_min and n>=0:
                self.actions.append(p)
            else:
                self.actions.append(-1)
        return
        



    def _get_obs(self):
        return self.memory
        
    def _get_info(self):
        return
        
    def step(self, action):
        # decay step
        self.memory = list(filter(lambda x:x>=0,map(lambda x:x-1,self.memory)))
        # generation attempt
        p = self.actions[action]
        success = np.random.binomial(n=1,p=p)
        if success == 1:
            self.memory.append(action)               
            self.memory.append(action)



        # An episode is done iff the agent has reached the target
        terminated = len(self.memory) >= self.num_qubits
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        truncated = False

        # purify the lowest
        if terminated == False:
            self.memory = sorted(self.memory)
            new_memory = []
            skip=False
            for i in range(len(self.memory)):
                if skip:
                    skip=False
                    continue
                elif self.memory[i]<=self.p_threshold_bin and i<len(self.memory)-1 and self.memory[i] == self.memory[i+1]:
                    fidelity = (self.threshold-1/4)*np.exp(self.memory[i]*self.decay_rate)+1/4
                    p_p = fidelity*fidelity+2*fidelity*(1-fidelity)/3+5*(1-fidelity)*(1-fidelity)/9
                    new_fidelity = (fidelity*fidelity+(1-fidelity)*(1-fidelity)/9)/p_p
                    n = np.floor(np.log((new_fidelity-0.25)/(self.threshold-0.25))/self.decay_rate).astype(int)
                    epl_success = success = np.random.binomial(n=1,p=p_p)
                    if epl_success:
                        new_memory.append(n)
                    skip=True
                else:
                    new_memory.append(self.memory[i])

            self.memory = new_memory

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

            #adaptive
            # lb = 0 if 0>self.bin_size_min else self.bin_size_min
            # if len(obs[-1])!=0 and action > np.min(obs[-1])-1 and np.min(obs[-1])>lb:
            #     action = np.min(obs[-1])-1


            observation, reward, terminated, truncated, info = self.step(action)
            obs.append(observation)
            # print(observation)
            if terminated:
                break
        return len(obs)
    
    def close(self):
        return
    


