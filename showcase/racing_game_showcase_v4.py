from games.racing_game import RacingGame
from lib.cma_es import CMA_ES_Active
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.activator_funcs import sigmoid
from lib.utilities import load_obj

#model trained on third map
cma_es = load_obj(700, 'models/cmaes_racing_v1')
game = RacingGame()

topology = [(4, sigmoid), (2, sigmoid)]
bestIndividual = None
bestScore = None

for i in range(50):
    individual = FixedTopologyNeuralNetwork(6, topology, cma_es.sample()[0])
    score = game.play(individual, map_id=3)
    if not bestScore or bestScore < score:
        bestScore = score
        bestIndividual = individual
    print(f'iter {i}, score {score}')

print('score on first map', game.play(bestIndividual, True, map_id=1))
print('score on second map', game.play(bestIndividual, True, map_id=2))
print('score on third map', game.play(bestIndividual, True, map_id=3))
