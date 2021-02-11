import gym
import time
import numpy as np
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
    
        col143 = np.nonzero(gameImage[:, 143])[0]
        col16 = np.nonzero(gameImage[:, 16])[0]
        
        p1 = (col143[0] + 1) / 160.0
        if col143[0] == 0:
            p1 = (col143[-1] - 14) / 160.0
        p1_b = (col143[-1] + 1) / 160.0

        p2 = (col16[0] + 1) / 160.0
        if col16[0] == 0:
            p2 = (col16[-1] - 14) / 160.0
        p2_b = (col16[-1] + 1) / 160.0
        
        tmp = np.nonzero(gameImage[:, 20:140])
        
        ballx = 0
        bally = 0

        if len(tmp[0]) != 0:
            ballx = (tmp[1][0] + 1) / 120.0
            bally = (tmp[0][0] + 1) / 160.0
            
            if tmp[0][0] == 0:
                bally = (tmp[0][-1] - 2) / 160.0

            self.scored = False


        ballby = bally + 3 / 160.0
        diffx = 0 if (ballx == 0 or self.prev_ballx == 0) else (ballx - self.prev_ballx)
        diffy = 0 if (bally == 0 or self.prev_bally == 0) else (bally - self.prev_bally)

        res = np.array([p1, p2, bally, 1 - ballx, diffy * 2, -diffx * 2])

        if ballx < 0.01 and ballx > 0: #we want to maximize enemy distance to ball when they miss
            if ballby < p2:
                self.distances += max(p2 - ballby, 0.05)
            if bally > p2_b:
                self.distances += max(bally - p2_b, 0.05)

        if ballx > 0.99 and ballx < 1: #and minimize our distance
            if ballby < p1:
                self.distances -= max(p1 - ballby, 0.05)
            if bally > p1_b:
                self.distances -= max(bally - p1_b, 0.05)


        self.prev_ballx = ballx
        self.prev_bally = bally

        return res

    def make_move_det(prob):
        if prob < 0.5: # np.random.uniform():
            return 3
        else:
            return 2
    
    def make_move_rand(prob):
        if prob < np.random.uniform():
            return 3
        else:
            return 2

    def play(self, network, render=False, wait=None, move_fun=make_move_rand):
        observation = self.env.reset()
        self.score = 0.0 
        self.prev_ballx = 0
        self.prev_bally = 0
        self.scored = False
        self.distances = 0

        #! Sometimes the game starts without the enemy for few frames, the loop below skips this part
        while True:
            if render:
                self.env.render()

            if self.processImage(observation)[0] != -1:
                break
            
            observation, reward, end, info = self.env.step(0)
        

        frames_played = 0

        while True:
            frames_played += 1

            if render:
                self.env.render()
            input = self.processImage(observation)

            prob = network.eval(input)
            move = move_fun(prob)

            observation, reward, end, info = self.env.step(move)

            if end:
                if render:
                    print("ended with stats:", self.score, frames_played / 1000, self.distances / 5)
                if reward == 1.0:
                    return 100 + self.score - frames_played / 1000.0 + self.distances / 5
                return self.score + frames_played / 1000.0 + self.distances / 5

            if wait:
                time.sleep(wait)
