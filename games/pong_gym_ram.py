import gym
import time
import numpy as np
import matplotlib.pyplot as plt

class PongGame:
    def __init__(self):
        self.env = gym.make('Pong-ram-v0')


    def make_move(self, prob):
        if np.random.uniform() < prob:
            return 2
        else:
            return 3
    
    def play(self, network, render=False, wait=None):
        observation = self.env.reset()
        score = 0.0
        frames_played = 0

        while True:
            frames_played += 1

            if render:
                self.env.render()
            input = np.array(observation) / 255.0

            prob = network.eval(input)
            #prob = prob / prob.sum()
            move = self.make_move(prob)

            observation, reward, end, info = self.env.step(move)
            score += reward

            if end:
                if score >= 0.0:
                    return 100.0 + score - frames_played / 1000.0
                return score + frames_played / 1000.0
            
            if wait is not None:
                time.sleep(wait)
