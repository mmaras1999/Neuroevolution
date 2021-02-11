from games.game_2048 import Game2048
from lib.utilities import load_obj
import pygame


pygame.init()
pygame.display.set_mode((1000, 1000))
pygame.display.set_caption('2048 NN')

neat = load_obj(3800, 'models/game2048/neat')
game = Game2048()

print(game.play(neat.bestFen, True, wait=0.2, moveDet=True))
