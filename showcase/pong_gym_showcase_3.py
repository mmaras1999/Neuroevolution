from games.pong_gym_v2 import PongGame
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import load_obj

#this model learn with fitness function that increased by the distance from enemy to ball (and decerases by bot distance to it) on each miss
topology = [(6, sigmoid), (3, sigmoid), (1, sigmoid)]
cma_es = load_obj(2000, 'models/cmaes_pong_v10')
game = PongGame()

#individual = cma_es.sample()[0]
best_id = 0
for i in range(len(cma_es.bestgens)):
    if cma_es.bestgens[i][1] < cma_es.bestgens[best_id][1]:
        best_id = i
individual = cma_es.bestgens[best_id][0]
print(best_id)

game.play(FixedTopologyNeuralNetwork(6, topology, individual), True)
