import gym
import time
import numpy as np
import matplotlib.pyplot as plt

class EnduroGame:
    def __init__(self):
        self.env = gym.make('Enduro-ram-v0')

    def make_move(self, prob=None):
        return np.random.choice(a=[0, 1, 2, 3, 4, 5, 6, 7, 8], p=prob)
    

    def play(self, network, render=False, wait=None):
        observation = self.env.reset()
        frames_played = 0
        reward_sum = 0

        for i in range(10):
            self.env.step(2)

        while True:
            frames_played += 1

            if render:
                self.env.render()
            input = np.array(observation)

            prob = network.eval(input)
            prob = prob / prob.sum()
            move = self.make_move(prob)

            observation, reward, end, info = self.env.step(move)
            reward_sum += reward
    
            if end:
                return 
            
            if wait is not None:
                time.sleep(wait)

game = EnduroGame()
print(game.play(None, True, 0.01))