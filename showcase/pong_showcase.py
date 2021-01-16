from games.pong import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

#the ball speed is increasing in time, the bot must first survive and then attack
#neural network bot is on top

topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)]
cma_es = load_obj(1000, 'models/cmaes_pong_v6')
    
game = PongGame()
individual = cma_es.sample()[0]
game.play_sample_game(FixedTopologyNeuralNetwork(6, topology, individual))
