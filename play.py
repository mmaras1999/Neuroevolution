from games.pong_gym_v2 import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

topology = [(2, sigmoid), (1, sigmoid)]
neat = load_obj(30, 'models/neat_pong_v2')


# print(XorGame().play(FixedTopologyNeuralNetwork(2, topology, cma_es.bestgens[23][0]),  render=True))
print(PongGame().play(neat.bestgens[23][2], render=True))

