from games.racing_game import RacingGame
from lib.neat import Neat
from lib.utilities import load_obj

#model trained on third map
neat = load_obj(200, 'models/car_racing/neat_race_v3')
game = RacingGame()
individual = neat.bestgens[-1][2]
print('score on first map', game.play(individual, True, map_id=1))
print('score on second map', game.play(individual, True, map_id=2))
print('score on third map', game.play(individual, True, map_id=3))
