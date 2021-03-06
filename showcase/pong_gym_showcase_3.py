from games.pong_gym_v2 import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

#this model was first learn on different pong game (first 1000 iterations, check cmaes_pong_v6) and then learn on this game
topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)]
cma_es = load_obj(530, 'models/pong_gym_v2/cmaes_pong_gym_rand_2')
    
game = PongGame()
individual = cma_es.bestgens[-1][0]
game.play(FixedTopologyNeuralNetwork(6, topology, individual), True, wait=0.01)