from games.pong import Game
from games.pong_gym import PongGame
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj
import numpy as np
import pickle


# 4 input neurons, 1 hidden layer with 4 neurons, 1 output neuron -- weights + biases = 25
topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)]
cma_es = load_obj(1000, 'models/cmaes_pong_v6')
game = Game()

game.play_with_NN(FixedTopologyNeuralNetwork(6, topology, cma_es.sample()[0]))
# game.play_with_bot()

