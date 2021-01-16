import pygame
import math
from enum import IntEnum
from pygame.locals import *

import numpy as np
import pickle

def botBob(state):
    myX, othX, bX, bY, bDX, bDY = state
    if bDY > 0:
        if myX < 400:
            return 1
        else:
            return -1
    
    if bX > 400:
        if myX - 30 < bX:
            return 1
        if myX - 30 > bX:
            return -1
        return 0
    else:
        if myX + 30 < bX:
            return 1
        if myX + 30 > bX:
            return -1
        return 0

class PongGame:
    class State(IntEnum):
        IN_PROGRES = 0
        WIN_TOP = 1
        WIN_BOT = 2
    
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption('Pong NN')
        self.board = pygame.Surface((800, 800)).convert()
        self.init_game()
    
    def init_game(self):
        self.state = PongGame.State.IN_PROGRES

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
        # print("game won by", new_state)
    
    def getGameState(self):
        return self.pads[0].pos.centerx, self.pads[1].pos.centerx, self.ball.pos.centerx, self.ball.pos.centery, self.ball.dx, self.ball.dy
    
    def getGameStateRev(self):
        return self.pads[1].pos.centerx, self.pads[0].pos.centerx, self.ball.pos.centerx, self.board.get_height() - self.ball.pos.centery, self.ball.dx, -self.ball.dy

  
    def play_two_bots(self, inp1, inp2, draw=False, MAX_ROUNDS=10**4):
        self.init_game()

        clock = pygame.time.Clock()

        # Event loop
        rounds = 0
        while self.state == PongGame.State.IN_PROGRES and rounds < MAX_ROUNDS:
            rounds += 1
            # for event in pygame.event.get():
            #     if event.type == QUIT:
            #         return

            # keys = pygame.key.get_pressed()
            
            self.pads[0].move(inp1(self.getGameState()))
            self.pads[1].move(inp2(self.getGameStateRev()))

            self.ball.move()
            # print(self.getGameState())

            if draw:
                clock.tick(100)
                self.draw(pygame.display.get_surface())
                pygame.display.flip()
            
        return rounds, self.state
    
    def play(self, NN):
        score = 0
        for i in range(10):
            rounds, win = self.play_two_bots(get_NN_bot(NN), botBob)
            if win == PongGame.State.WIN_TOP:
                score += 30000 - rounds + abs(self.pads[1].pos.centerx - self.ball.pos.centerx)
            if win == PongGame.State.WIN_BOT:
                score += rounds - abs(self.pads[0].pos.centerx - self.ball.pos.centerx)
            if win == PongGame.State.IN_PROGRES:
                score += 5000 + rounds
        return score / (10**4) / 10

    def play_sample_game(self, NN):
        self.play_two_bots(get_NN_bot(NN), botBob, draw=True, MAX_ROUNDS=2000)

    def play_with_bot(self, bot=botBob):
        self.__init__()

        clock = pygame.time.Clock()

        # Event loop
        rounds = 0
        while self.state == PongGame.State.IN_PROGRES and rounds < 10**4:
            rounds += 1
            for event in pygame.event.get():
                if event.type == QUIT:
                    return

            keys = pygame.key.get_pressed()
            
            self.pads[1].move(inp1(keys))
            self.pads[0].move(bot(self.getGameState()))

            self.ball.move()

            clock.tick(30)
            self.draw(pygame.display.get_surface())
            pygame.display.flip()
            
        return rounds, self.state

    def play_with_NN(self, NN):
        self.play_with_bot(get_NN_bot(NN))


class Ball:
    def __init__(self, game):
        self.pos = pygame.Rect(game.board.get_width() / 2, game.board.get_height() / 2, 25, 25)
        self.angle = math.pi / 2
        self.speed = 15
        self.game = game
        self.dx = math.cos(self.angle) * self.speed
        self.dy = -math.sin(self.angle) * self.speed

    def move(self):
        self.dx = math.cos(self.angle) * self.speed
        self.dy = -math.sin(self.angle) * self.speed
        self.pos = self.pos.move(self.dx, self.dy)
        self.speed += 1 / 500

      
        #bounce from map borders
        if self.pos.left < 0:   
            self.pos.left = -self.pos.left
            self.angle = math.atan2(math.sin(self.angle), -math.cos(self.angle))

        if self.pos.right > self.game.board.get_width():
            self.pos.right = 2 * self.game.board.get_width() - self.pos.right
            self.angle = math.atan2(math.sin(self.angle), -math.cos(self.angle))

        #bounce from pads
        if self.pos.top < self.game.pads[0].pos.bottom:
            diff = (self.pos.centerx - self.game.pads[0].pos.centerx) / (self.game.pads[0].pos.width + self.pos.width) * 2

            if diff > 1 or diff < -1:
                self.game.setState(PongGame.State.WIN_BOT)
            else:
                self.angle = math.pi * 3 / 2 + math.pi / 3 * diff
                self.pos.top += 2 * (self.game.pads[0].pos.bottom - self.pos.top)

        if self.pos.bottom > self.game.pads[1].pos.top:
            diff = (self.pos.centerx - self.game.pads[1].pos.centerx) / (self.game.pads[1].pos.width + self.pos.width) * 2

            if diff > 1 or diff < -1:
                self.game.setState(PongGame.State.WIN_TOP)
            else:
                self.angle = math.pi / 2 - math.pi / 3 * diff
                self.pos.bottom -= 2 * (self.pos.bottom - self.game.pads[1].pos.top)


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
        self.pos = self.pos.move(dir * 11, 0)
        
        if self.pos.centerx < 0:
            self.pos.centerx = 0
        
        if self.pos.centerx > self.game.board.get_width():
            self.pos.centerx = self.game.board.get_width()

def inp1(keys):
    if keys[K_LEFT]:
        return -1
    if keys[K_RIGHT]:
        return 1
    return 0

def inp2(keys):
    if keys[K_a]:
        return -1
    if keys[K_d]:
        return 1
    return 0

def bot(state):
    myX, othX, bX, bY, bDX, bDY = state
    if bX > 400:
        if myX - 30 < bX:
            return 1
        if myX - 30 > bX:
            return -1
        return 0
    else:
        if myX + 30 < bX:
            return 1
        if myX + 30 > bX:
            return -1
        return 0


def get_NN_bot(NN):
    def make_move(prob):
        if np.random.uniform() > prob:
            return 1
        else:
            return -1
    def fun(state):
        state = np.array(state)
        state[:4] /= 800 #positions
        state[4:6] /= 35 #speed
        return make_move(NN.eval(state))
        
    return fun