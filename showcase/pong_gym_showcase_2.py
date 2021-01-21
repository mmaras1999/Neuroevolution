from games.pong_gym_v2 import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

#this model learn with fitness function that increased by the distance from enemy to ball (and decerases by bot distance to it) on each bounce and miss
topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)]
cma_es = load_obj(1050, 'models/cmaes_pong_v9')
    
game = PongGame()
individual = cma_es.sample()[0]
game.play(FixedTopologyNeuralNetwork(6, topology, individual), True, wait=0.01)
