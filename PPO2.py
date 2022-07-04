import gym


class BasicEnv(gym.Env):
    def __init__(self, n_pits):
        self.action_space = gym.spaces.Discrete(n_pits)
        self.observation_space = gym.spaces.Discrete(n_pits)

    def step(self, action):
        state = 1

        if action:
            reward = 1
        else:
            reward = -1

        done = True
        info = {}
        return state, reward, done, info


    def reset(self):
        state = 0
        return state