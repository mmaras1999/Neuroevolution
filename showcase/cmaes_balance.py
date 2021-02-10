from games.balance_game import BalanceGame
from lib.utilities import load_obj
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
import sys

sampling_counts = 1
if len(sys.argv) > 1:
    sampling_counts = int(sys.argv[1])

bestIndiv = None
bestIndivScore = None
model = load_obj(800, 'models/cmaes_balance_v3') 
game = BalanceGame()
topology = [(4, sigmoid), (1, sigmoid)]

for i in range(sampling_counts):
    indiv = model.sample()[0]
    score = game.play(FixedTopologyNeuralNetwork(4,topology, indiv))
    if not bestIndivScore or bestIndivScore < score:
        bestIndivScore = score
        bestIndiv = indiv


print(game.play(FixedTopologyNeuralNetwork(4,topology, bestIndiv), render=True, games_amount=3))

