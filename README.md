## Path planning in grid world using Markov Decision Process and Reinforcement Learning

This repository contains code for path planning using Value Iteration, Policy Iteration and Q learning.
Markov Decision Process (MDP) Toolbox for Python is little modification for analysis is used for Reinforcement Learning Algorithms.
Full analysis is presented in the analysis paper. Conclusions for comparing the algorithms and their performance is presented in the end.

Parameters of the grid world can be varied. Following are the parameters of grid world.
  - maxRow
  - maxCol
  - num_obstacle_pts
  - cor_pr
  - wr_pr
  - n_actions
  - startRow
  - startCol
  - goalRow 
  - goalCol
  - goalReward
  - obstReward
  - stayReward
  - gamma

# Files in the repository
 - README.md
 - mdp.py -- modified version of Qlearning, Policy Iteration and Value Iteration for better analysis.
 - mdp1_v1.py -- file run analysis using Policy Iteration and Value Iteration algorithm, it saves all the data, plots and graphs.
 - mdp1_v3.py -- file run analysis using Q-Learning Iteration algorithm, it saves all the data, plots and graphs.
 - mdptoolbox/ -- supporting files for mdp.py from MDP Toolbox
 - sagarwal311-analysis.pdf -- Analysis paper	

<center>![Alt Text](https://i.imgflip.com/2clnan.gif)</center>
