from games.xor_game import XorGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

topology = [(2, sigmoid), (1, sigmoid)]
cma_es = load_obj(100, 'models/cmaes_xor_v1')


print(XorGame().play(FixedTopologyNeuralNetwork(2, topology, cma_es.bestgens[-1][0]),  render=True))

