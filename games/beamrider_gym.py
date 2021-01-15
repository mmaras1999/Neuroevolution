import gym
import time
import numpy as np
import matplotlib.pyplot as plt

class BeamRiderGame:
    def __init__(self):
        self.env = gym.make('BeamRider-ram-v0')

    #* Available moves:
    #* 1 - fire
    #* 2 - special
    #* 3 - right
    #* 4 - left
    #? 5 - right and special
    #? 6 - left and special
    #? 7 - right and fire
    #? 8 - left and fire
    def make_move(self, prob):
        return np.random.choice(a=[1, 2, 3, 4], p=prob)
    

    def play(self, network, render=False, wait=None):
        observation = self.env.reset()
        frames_played = 0

        while True:
            frames_played += 1

            if render:
                self.env.render()
            input = np.array(observation)

            prob = network.eval(input)
            prob = prob / prob.sum()
            move = self.make_move(prob)

            observation, reward, end, info = self.env.step(move)
    
            if end:
                return reward * 10 + frames_played / 10.0
            
            if wait is not None:
                time.sleep(wait)
