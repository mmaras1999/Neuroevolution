from random import randint, seed
from pygame.locals import *
import numpy as np
import pygame
import time

class Game2048:
    ### initializes the 2048 Game class
    def __init__(self, random_seed=None):
        
        if random_seed is not None:
            seed(random_seed)

        self.fontBig = pygame.font.SysFont('robotobold', 128)
        self.fontNorm = pygame.font.SysFont('robotoextrabold', 48)
        self.colors = [(255, 255, 255), 
                       (255, 240, 230),
                       (255, 224, 204),
                       (255, 209, 179),
                       (255, 194, 153),
                       (255, 179, 128),
                       (255, 163, 102),
                       (255, 148, 77),
                       (255, 133, 51),
                       (255, 117, 26),
                       (255, 102, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0),
                       (230, 92, 0)]
    
    def initMap(self):
        self.lastMove = -1
        self.board = [[None for i in range(4)] for i in range(4)]
        self.place_random()


    ### check if move is valid
    def check_valid_move(self, move):

        if move == 0:
            for row in range(1, 4):
                for column in range(0, 4):
                    if (self.board[row][column] is not None):
                        if (self.board[row - 1][column] is None or 
                            self.board[row - 1][column] == self.board[row][column]):
                            return True

        if move == 1:
            for row in range(3):
                for column in range(4):
                    if (self.board[row][column] is not None):
                        if (self.board[row + 1][column] is None or 
                            self.board[row + 1][column] == self.board[row][column]):
                            return True

        if move == 2:
            for row in range(4):
                for column in range(1, 4):
                    if (self.board[row][column] is not None):
                        if (self.board[row][column - 1] is None or
                            self.board[row][column - 1] == self.board[row][column]):
                            return True
        
        if move == 3:
            for row in range(4):
                for column in range(3):
                    if (self.board[row][column] is not None):
                        if (self.board[row][column + 1] is None or
                            self.board[row][column + 1] == self.board[row][column]):
                            return True

        return False

    ### performs a move
    def update_move(self, move):
        
        if move == 0:
            for row in range(1, 4):
                for col in range(4):
                    if self.board[row][col] is not None:
                        a = row
                        b = col

                        while a > 0:
                            if self.board[a - 1][b] is None:
                                self.board[a - 1][b] = self.board[a][b]
                                self.board[a][b] = None
                                a -= 1
                            elif self.board[a - 1][b] == self.board[a][b]:
                                self.board[a - 1][b] += self.board[a][b]
                                self.board[a][b] = None
                                break
                            else:
                                break

        if move == 1:
            for row in range(3, -1, -1):
                for col in range(4):
                    if (self.board[row][col] is not None):
                        a = row
                        b = col

                        while a < 3:
                            if (self.board[a + 1][b] is None):
                                self.board[a + 1][b] = self.board[a][b]
                                self.board[a][b] = None
                                a += 1
                            elif self.board[a + 1][b] == self.board[a][b]:
                                self.board[a + 1][b] += self.board[a][b]
                                self.board[a][b] = None
                                break
                            else:
                                break

        if move == 2:
            for row in range(4):
                for col in range(1, 4):
                    if (self.board[row][col] is not None):
                        a = row
                        b = col

                        while b > 0:
                            if (self.board[a][b - 1] is None):
                                self.board[a][b - 1] = self.board[a][b]
                                self.board[a][b] = None
                                b -= 1
                            elif self.board[a][b - 1] == self.board[a][b]:
                                self.board[a][b - 1] += self.board[a][b]
                                self.board[a][b] = None
                                break
                            else:
                                break
        
        if move == 3:
            for row in range(4):
                for col in range(3, -1, -1):
                    if (self.board[row][col] is not None):
                        a = row
                        b = col

                        while b < 3:
                            if (self.board[a][b + 1] is None):
                                self.board[a][b + 1] = self.board[a][b]
                                self.board[a][b] = None
                                b += 1
                            elif (self.board[a][b + 1] == self.board[a][b]):
                                self.board[a][b + 1] += self.board[a][b]
                                self.board[a][b] = None
                                break
                            else:
                                break

    ### calculate the score
    def calc_score(self):
        score = 0
        
        for row in self.board:
            for val in row:
                if val is not None:
                    score += val

        return score

    ### checks if player lost
    def lost(self):
        lost = True

        for mv in [0, 1, 2, 3]:
            if self.check_valid_move(mv):
                lost = False

        return lost
    
    ### place random tile
    def place_random(self):

        ### get possible positions
        possible_pos = []

        for row in range(4):
            for col in range(4):
                if (self.board[row][col] is None):
                    possible_pos.append([row, col])
        
        if len(possible_pos) == 0:
            return
        
        position = possible_pos[randint(0, len(possible_pos) - 1)]

        r = randint(0, 100)

        self.board[position[0]][position[1]] = 2 if (r <= 75) else 4

    ### tries to perform a move, returns a score if lost, None otherwise
    def make_move(self, prob):
        move = np.random.choice(a=[0, 1, 2, 3], p=prob)
        #move = np.argmax(prob)
        self.lastMove = move

        ## check if move is valid
        if not self.check_valid_move(move):
            return self.calc_score()
        
        ## update the board
        self.update_move(move)

        ## make a new random tile
        self.place_random()

        ## check if lost
        if self.lost():
            return self.calc_score()

        return None

    def play(self, network, render=False, wait=None):
        score = 0
        for _ in range(10 if not render else 1):
            self.initMap()

            while(True):

                if render:
                    self.display(pygame.display.get_surface())
                    pygame.display.flip()

                input = np.array(self.board).ravel()
                input[input == None] = 1.0
                input = np.log2(input.astype(float))
                input = np.concatenate((input, [self.lastMove]))

                prob = network.eval(input)
                prob += 0.0001
                # prob = np.array([0.25, 0.25, 0.25, 0.25])
                prob = prob / prob.sum()
                res = self.make_move(prob)

                if res is not None:
                    score += res
                    break

                if wait is not None:
                    time.sleep(wait)
        return score / 10

    def display(self, screen):
        screen.fill((255, 153, 102))
        pygame.draw.rect(screen, (255, 204, 153), (100, 150, 800, 800))

        nums = np.array(self.board).ravel()
        nums[nums == None] = 1.0
        nums = nums.astype(float)
        nums = np.log2(nums).astype(int)

        title = self.fontBig.render('2048 Game', True, (255, 230, 230))
        screen.blit(title, (275, 50))


        for i in range(4):
            for j in range(4):
                
                pygame.draw.rect(screen, self.colors[int(nums[4 * i + j])], (50 + 75 + i * 190, 175 + j * 190, 180, 180))
                x = 50 + 75 + i * 190
                y =  175 + j * 190
                
                if nums[4 * i + j] != 0:
                    bzzt = self.fontNorm.render('{0}'.format(2 ** nums[4 * i + j]), True, (75, 75, 75))
                    rct = bzzt.get_rect(center=(x + 90, y + 90))
                    screen.blit(bzzt, rct)
