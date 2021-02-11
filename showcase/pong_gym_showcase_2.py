from games.pong_gym_v2 import PongGame
from lib.utilities import load_obj
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
import sys

sampling_counts = 0
if len(sys.argv) > 1:
    sampling_counts = int(sys.argv[1])

bestIndiv = None
bestIndivScore = None
model = load_obj(500, 'models/cmaes_pong_v11') 
topology = [(3, sigmoid), (1, sigmoid)]

for i in range(sampling_counts):
    indiv = model.sample()[0]
    game = PongGame()
    score = game.play(FixedTopologyNeuralNetwork(6 ,topology, indiv), move_fun=PongGame.make_move_det)
    if not bestIndivScore or bestIndivScore < score:
        bestIndivScore = score
        bestIndiv = indiv

game = PongGame()
print(game.play(FixedTopologyNeuralNetwork(6, topology, model.sample()[0]), render=True, move_fun=PongGame.make_move_det, wait=0.01))

