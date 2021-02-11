## Neuroevolution
##### Project for evolutionary algorithms classes.
#### *Authors: Michał Maras, Michał Kępa*

### Description
The main goal of the project is to implement most popular neuroevolution algorithms and test their performance on various video games.


### Requirements
The following libraries are necessary to run the project:
* NumPy v.1.19.0
* SciPy v.1.5.2
* OpenAI Gym v.0.18.0
* OpenAI Gym Atari v.0.2.6
* MatPlotLib v.3.3.2
* Pygame v.2.0.1

The project was tested on Python3 v.3.8.5

For better code readability we suggest using VS Code with Better Comments addon

### How to run

1. Go to the project's main directory
2. Run setup.sh script using the following command: ```. ./setup.sh```
3. You can preview model's performance by running scripts in *showcase* directory. You can train new or existing models using scripts in *train* directory.

### Plotting

Each model stores history of fitness values. You can plot them using *plot.py* script for LM-MA-ES, MA-ES and CMA-ES and *plot_neat.py* for NEAT - modify model's number and path in load_obj function to chose a particular model.