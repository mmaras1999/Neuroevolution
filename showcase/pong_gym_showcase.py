from games.pong_gym import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj


topology = [(6, sigmoid), (1, sigmoid)]
cma_es = load_obj(4700, 'models/pong_gym/cmaes_pong_gym_rand_3')
    
game = PongGame()
individual = cma_es.sample()[0]
game.play(FixedTopologyNeuralNetwork(6, topology, individual), True)
