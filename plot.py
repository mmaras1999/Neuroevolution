
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lib.utilities import load_obj

#check best[644] (645 population), had -14 score v10
#check best[1163] (1164 population), had -15.8 score v10
model = load_obj(200, 'models/neat_race_v3') 

scores = []
for gen, scr, fen in model.bestgens:
    scores.append(scr)

plt.plot(scores, label='best')
aver = 30
plt.plot(range(aver-1, len(scores)), np.convolve(scores, np.ones(aver) / aver, mode='valid'), label=f'moving best avg {aver}')

plt.plot(model.genmeans, label='mean')
aver = 30
plt.plot(range(aver-1, len(model.genmeans)), np.convolve(model.genmeans, np.ones(aver) / aver, mode='valid'), label=f'moving mean avg {aver}')
plt.legend()
plt.show()

