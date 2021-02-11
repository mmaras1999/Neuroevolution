from games.pong_gym_v2 import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj


topology = [(3, sigmoid), (1, sigmoid)]
cma_es = load_obj(500, 'models/cmaes_pong_v11')
    
game = PongGame()
individual = cma_es.bestgens[-1][0]
game.play(FixedTopologyNeuralNetwork(6, topology, individual), True, wait=None, move_fun=PongGame.make_move_det)
