from games.pong import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

#here one can play with neutral network bot, use arrow to move the pad
#neural network bot is on top

topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)]
cma_es = load_obj(1000, 'models/cmaes_pong_v6')
    
game = PongGame()
individual = cma_es.sample()[0]
game.play_with_NN(FixedTopologyNeuralNetwork(6, topology, individual))
