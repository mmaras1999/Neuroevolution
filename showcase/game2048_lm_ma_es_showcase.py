from games.game_2048 import Game2048
from lib.cma_es import CMA_ES_Active
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.activator_funcs import sigmoid
from lib.utilities import load_obj
import pygame

pygame.init()
pygame.display.set_mode((1000, 1000))
pygame.display.set_caption('2048 NN')

lm_ma_es = load_obj(5800, 'models/game2048/lm_ma_es_v3')
game = Game2048()

topology = [(17, sigmoid), (4, sigmoid)]
individual = lm_ma_es.sample()[0]
print(game.play(FixedTopologyNeuralNetwork(17, topology, individual), True, wait=0.2))
