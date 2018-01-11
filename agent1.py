import gym
import fxenv

env = gym.make('FxEnv-v1')

env.reset()


finished = False
reward_total = 0

while not(finished) :
    _, reward, finished, _ = env.step(0)
    reward_total += reward

print(reward_total)
