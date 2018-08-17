# DRL_P1_Navigation
Udacity Deep Reinforcement Learning P1 on Navigation
Project Details

This project uses Unity Machine Learning Agents (ML-Agents) to train a Deep Reinforcement Learning algo which chooses yellow bananas instead of blue banana.

The environment state space of size 37 and an action space of size 4 (up, down, right, left).  Each episode was capped at 500 steps.  Epsilon began at .7 on the first episode and decayed by 0.995 in each subsequent episode up to a minimum of 0.1.

The score is +1 for a yellow banana and -1 for a blue banana.  The goal is to collect a score of +13 over 100 steps.

Source and Dependencies:
This repository was originally sourced from https://github.com/udacity/deep-reinforcement-learning#dependencies
To run this depository I downlowded the Unity app Banana from https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip

Relative to the original settings I changed the following parameters in Agent.py:
BUFFER_SIZE = int(1e5)  # replay buffer size 1e5, same as original
BATCH_SIZE = 128         # minibatch size, original = 64
GAMMA = 0.99            # discount factor, same as original
TAU = .0055             # for soft update of target parameters, original = 1e-3
LR = 5.5e-3               # learning rate, original = 5e-4
UPDATE_EVERY = 10        # how often to update the network, original = 4

This code was run via a jupyter notebook using the Udacity DRLND conda environment.

To train the agent run via Conrad_P1.ipynb.  Conrad_P1.ipynb will call on p1_agent.py for the Unity Agent.  

The Agent will train on a neural network specified in p1_model.py.  This network has an input layer of dimension 37 equal to the state space, then two fully connected layers of width 64 which both use relu activation on the forward pass, and finally a 4 element output layer, one for each action in the q-table.



