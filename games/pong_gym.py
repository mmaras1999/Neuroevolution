<<<<<<< HEAD
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
=======
import pygame
import math
from enum import IntEnum
from pygame.locals import *

class Game:
    class State(IntEnum):
        IN_PROGRES = 0
        WIN_TOP = 1
        WIN_BOT = 2
    
    def __init__(self):
        self.state = Game.State.IN_PROGRES
        self.board = pygame.Surface((800, 800)).convert()

        self.ball = Ball(self)
        self.pads = [Pad(self, 0), Pad(self, self.board.get_height())]

    def draw(self, screen):
        self.board.fill((127, 127, 127))

        for pad in self.pads:
            pad.draw(self.board)
        self.ball.draw(self.board)

        screen.blit(self.board, (100,100))

    def setState(self, new_state):
        self.state = new_state
        print("game won by", new_state)
    
    def getGameState(self):
        return self.pads[0].pos.centerx, self.pads[1].pos.centerx, self.ball.pos.centerx, self.ball.pos.centery
    
    def getGameStateRev(self):
        return self.pads[1].pos.centerx, self.pads[0].pos.centerx, self.ball.pos.centerx, self.board.get_height() - self.ball.pos.centery


class Ball:
    def __init__(self, game):
        self.pos = pygame.Rect(game.board.get_width() / 2, game.board.get_height() / 2, 25, 25)
        self.angle = math.pi / 3
        self.speed = 15
        self.game = game

    def move(self):
        self.pos = self.pos.move(math.cos(self.angle) * self.speed, -math.sin(self.angle) * self.speed)
      
        if self.pos.left < 0:   
            self.pos.left = -self.pos.left
            self.angle = math.atan2(math.sin(self.angle), -math.cos(self.angle))

        if self.pos.right > self.game.board.get_width():
            self.pos.right = 2 * self.game.board.get_width() - self.pos.right
            self.angle = math.atan2(math.sin(self.angle), -math.cos(self.angle))

        if self.pos.top < self.game.pads[0].pos.bottom:
            diff = (self.pos.centerx - self.game.pads[0].pos.centerx) / (self.game.pads[0].pos.width / 2)

            if diff > 1 or diff < -1:
                self.game.setState(Game.State.WIN_BOT)
            else:
                # self.angle = math.atan2(-math.sin(self.angle), math.cos(self.angle))
                self.angle = math.pi * 3 / 2 + math.pi / 3 * diff
                self.pos.top += 2 * (self.game.pads[0].pos.bottom - self.pos.top)

        if self.pos.bottom > self.game.pads[1].pos.top:
            diff = (self.pos.centerx - self.game.pads[1].pos.centerx) / (self.game.pads[1].pos.width / 2)

            if diff > 1 or diff < -1:
                self.game.setState(Game.State.WIN_TOP)
            else:
                # self.angle = math.atan2(-math.sin(self.angle), math.cos(self.angle))
                self.angle = math.pi / 2 - math.pi / 3 * diff
                self.pos.bottom -= 2 * (self.pos.bottom - self.game.pads[1].pos.top)

        # print(self.pos, self.angle)

    def draw(self, screen):
        pygame.draw.rect(screen, (255,255,255), self.pos)


class Pad:
    def __init__(self, game, y):
        self.pos = pygame.Rect(0, 0, 200, 50)
        self.pos.center = (game.board.get_width() / 2, y)
        self.game = game

    def draw(self, screen):
        pygame.draw.rect(screen, (200, 200, 200), self.pos)

    def move(self, dir):
        self.pos = self.pos.move(dir * 10, 0)


def main(inp1, inp2, draw=False, MAX_ROUNDS=10**4):
    # Initialise screen

    # Fill background

    # Display some text
    # font = pygame.font.Font(None, 36)
    # text = font.render("Hello There", 1, (10, 10, 10))
    # textpos = text.get_rect()
    # textpos.centerx = background.get_rect().centerx
    # background.blit(text, textpos)

    # Blit everything to the screen

    clock = pygame.time.Clock()
    game = Game()

    # Event loop
    rounds = 0
    while game.state == Game.State.IN_PROGRES and rounds < MAX_ROUNDS:
        rounds += 1
        for event in pygame.event.get():
            if event.type == QUIT:
                return

        # keys = pygame.key.get_pressed()
        
        game.pads[0].move(inp1(game.getGameState()))
        game.pads[1].move(inp2(game.getGameStateRev()))

        game.ball.move()
        print(game.getGameState())

        if draw:
            clock.tick(30)
            game.draw(pygame.display.get_surface())
            pygame.display.flip()
    return rounds, game.state


def inp1(state, keys):
    if keys[K_LEFT]:
        return -1
    if keys[K_RIGHT]:
        return 1
    return 0

def inp2(state, keys):
    if keys[K_a]:
        return -1
    if keys[K_d]:
        return 1
    return 0

def bot(state):
    myX, othX, bX, bY = state
    if myX < bX - 50:
        return 1
    if myX > bX + 50:
        return -1
    return 0

def teach():
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption('Pong NN')


    print(main(bot, bot, False))
    pass


if __name__ == '__main__': teach()
>>>>>>> aeea99b46c93de8e5cc27a55f783b8e828e8fc81
