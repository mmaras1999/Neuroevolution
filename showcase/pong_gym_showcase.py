from games.pong_gym import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj


topology = [(6, sigmoid), (1, sigmoid)]
cma_es = load_obj(3300, 'models/cmaes_pong_v8')
    
game = PongGame()
individual = cma_es.sample()[0]
<<<<<<< HEAD
game.play(FixedTopologyNeuralNetwork(6, topology, individual), True, wait=0.03)
=======
game.play(FixedTopologyNeuralNetwork(6, topology, individual), True, wait=0.01)
>>>>>>> 8432a14dccf880826e70c35bcbac2ee93c6315ce
