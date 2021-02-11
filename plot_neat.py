# script for plotting performance plots for NEAT
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.utilities import load_obj


#model = load_obj(500, 'models/cmaes_pong_v11') 
#scores = model.bestgens

model = load_obj(100, 'models/neat_pong_v1')
scores = [] 
for gen, scr, tmp in model.bestgens:
    scores.append(scr) 

plt.plot(scores, label='best')
plt.plot(model.genmeans, label='mean')

aver = 20
plt.plot(range(aver-1, len(scores)), np.convolve(scores, np.ones(aver) / aver, mode='valid'), label=f'moving best avg {aver}')
plt.plot(range(aver-1, len(model.genmeans)), np.convolve(model.genmeans, np.ones(aver) / aver, mode='valid'), label=f'moving mean avg {aver}')
plt.legend()
#plt.savefig('2048_neat.png')
plt.show()

