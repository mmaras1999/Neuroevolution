import gym
import numpy as np
from lib.fixed_top_nn import FixedTopologyNeuralNetwork

class PongGame:
    def __init__(self):
        self.env = gym.make('Pong-v0')

    def processImage(self, gameImage):
        pass

    def make_move(self, prob):
        if np.random.uniform() > prob:
            return 2
        else:
            return 3
    

    def play(network):
        observation = env.reset()

        while True:
            env.render()
            input = processImage(observation)

            move = make_move(network.eval(input))

            observation_image, reward, end, info = env.step(move)
    
            if end:
                return reward
