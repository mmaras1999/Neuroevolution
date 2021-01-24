import gym
import time
import numpy as np
import matplotlib.pyplot as plt

class BalanceGame:
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def make_move_det(prob):
        if 0.5 < prob:
            return 0
        else:
            return 1
    def make_move_rand(prob):
        if np.random.uniform() < prob:
            return 0
        else:
            return 1
    

    def play(self, network, render=False, wait=None, move_fun=make_move_det, games_amount=10):
        result = 0
        for _ in range(games_amount):
            observation = self.env.reset()      
            score = 0
            frames_played = 0

            while True:
                frames_played += 1
                observation[:2] /= 4.8
                observation[2:] /= 0.418

                if render:
                    self.env.render()

                prob = network.eval(observation)
                move = move_fun(prob)

                observation, reward, end, info = self.env.step(move)
                score += 0.5 - abs(observation[0] / 4) - abs(observation[2] / 4)
        
                if end:
                    result +=  frames_played + score
                    break
                
                if wait is not None:
                    time.sleep(wait)
        
        return result / games_amount