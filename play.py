from games.pong import Game
from games.pong_gym import PongGame
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
import numpy as np
import pickle

def load_obj(gen):
    file = open('models/cma_es{0}.obj'.format(gen), 'rb') 
    obj = pickle.load(file)
    file.close()
    return obj


# 4 input neurons, 1 hidden layer with 4 neurons, 1 output neuron -- weights + biases = 25
topology = [(4, sigmoid), (1, sigmoid)]
cma_es = load_obj(300)
game = Game()

game.play_with_NN(FixedTopologyNeuralNetwork(4, topology, cma_es.sample()[0]))

