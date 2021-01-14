import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib.fixed_top_nn import FixedTopologyNeuralNetwork

class PongGame:
    def __init__(self):
        self.env = gym.make('Pong-v0')

    def processImage(self, gameImage):
        #cut scores
        gameImage = gameImage[35:194]
        #erase background
        gameImage[gameImage == 144] = 0
        gameImage[gameImage == 72] = 0
        gameImage[gameImage == 17] = 0

        # player 1: column 140-143
        # player 2: column 16-19
        # ball: columns 29-140

        #find paddles and ball, construct an np array
        if(len(np.unique(gameImage[:, 16])) != 4):
            return np.array([-1])

        if(len(np.nonzero(gameImage[:, 0:15])[0]) != 0 and not self.scored):
            self.score += 1.0
            self.scored = True
        if(len(np.nonzero(gameImage[:, 144:159])[0]) != 0 and not self.scored):
            self.scored = True
            self.score -= 1.0
    
        p1 = np.nonzero(gameImage[:, 143])[0][0] / 160.0
        p2 = np.nonzero(gameImage[:, 16])[0][0] / 160.0
        
        tmp = np.nonzero(gameImage[:, 20:140])
        
        ballx = 0
        bally = 0

        if len(tmp[0]) != 0:
            ballx = tmp[1][0] / 120.0
            bally = tmp[0][0] / 160.0
            self.scored = False
        
        return np.array([p1, p2, ballx, bally])


    def make_move(self, prob):
        if np.random.uniform() > prob:
            return 3
        else:
            return 2
    

    def play(self, network):
        observation = self.env.reset()
        self.score = 0.0 
        self.scored = False

        #! Sometimes the game starts without the enemy for few frames, the loop below skips this part
        while True:
            self.env.render()

            if self.processImage(observation)[0] != -1:
                break
            
            observation, reward, end, info = self.env.step(0)
        

        frames_played = 0

        while True:
            frames_played += 1

            self.env.render()
            input = self.processImage(observation)

            move = self.make_move(network.eval(input))

            observation, reward, end, info = self.env.step(move)
    
            if end:
                if reward == 1.0:
                    return self.score - frames_played / 1000.0
                return self.score + frames_played / 1000.0
