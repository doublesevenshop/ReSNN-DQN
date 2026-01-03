import flappy_bird_gymnasium

import gymnasium as gym


env = gym.make("FlappyBird-v0")

print("Obs High:", env.observation_space.high)
print("Obs Low:", env.observation_space.low)