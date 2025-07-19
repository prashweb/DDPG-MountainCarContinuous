# DDPG-MountainCarContinuous

This repository contains an implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the [MountainCarContinuous-v0](https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/) environment from OpenAI Gym.

## Overview

The aim of this project is to demonstrate how DDPG, an actor-critic, model-free algorithm based on deep reinforcement learning, can be applied to classic continuous control tasks like MountainCarContinuous-v0.

<p align="center">
  <img src="https://gymnasium.farama.org/_images/mountain_car_continuous.gif" alt="MountainCarContinuous-v0" width="400"/>
</p>

## Features

- DDPG implementation using PyTorch (or TensorFlow if you used it).
- Continuous action space policy for effective control of the car.
- Script for training and evaluation.
- Configurable hyperparameters.

## Requirements

- Python 3.6+
- numpy
- gym (or gymnasium)
- torch (if using PyTorch)
- matplotlib (for plotting, optional)
