# SpikingDeepRLControl
This repository serves as the collaboration of practical project computational neuro engineering NST, TUM.

## Goal
The main goal of this task is to make the robot arm behave properly to reach a pregiven position by applying spiking neural network based deep reinforcement learning algorithm.

However, some high level goal could also be possible after the achievment e.g. detect and push objects to a given region or more generally take the images as the state (spiking CNN).

## Approach
The basic ideas come from the paper published by Google DeepMind Playing Atari with Deep Reinforcement Learning
(https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and Continuous Control with deep reinforcement learning(https://arxiv.org/pdf/1509.02971.pdf).

More specificly, our approach is to apply spiking neural network as the action-value approximator in a standard Q-learning framework. This idea will be firstly verified in the mathematical emulator and transplanted to the real robot afterwards.

## Usage
git clone git@github.com:356255531/SpikingDeepRLControl.git
cd SpikingDeepRLControl
make emulator (robot_arm) # if you want run it an real robots

## Contributor
Zhiwei Han, Bo Huang, Meng Wang.