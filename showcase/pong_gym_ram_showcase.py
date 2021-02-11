from games.pong_gym_ram import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

topology = [(64, sigmoid), (32, sigmoid), (1, sigmoid)]
lm_ma_es = load_obj(15900, 'models/pong_gym_ram/lm_ma_es_pong_ram')
    
game = PongGame()
individual = lm_ma_es.sample()[0]
game.play(FixedTopologyNeuralNetwork(128, topology, individual), True, wait=0.01)
