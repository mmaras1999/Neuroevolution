from games.pong import Game
from games.pong_gym import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

# 4 input neurons, 1 hidden layer with 4 neurons, 1 output neuron -- weights + biases = 25
topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)]
cma_es = load_obj(530, 'models/cmaes_pong_v10')


print(PongGame().play(FixedTopologyNeuralNetwork(6, topology, cma_es.bestgens[-1][0]), render=True))
# game.play_with_bot()

