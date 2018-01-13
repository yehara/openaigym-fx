import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import rl.callbacks

import fxenv

ENV_NAME = 'FxEnv-v1'
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=100,
               target_model_update=1e-2,
               policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

nb_pre_steps = 0
nb_steps = 14400

if nb_pre_steps > 0:
    dqn.load_weights('weights/dqn_{}_weights_{}.h5f'.format(ENV_NAME, nb_pre_steps))

if nb_steps > 0:
    dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=1, log_interval=1440)
    dqn.save_weights('weights/dqn_{}_weights_{}.h5f'.format(ENV_NAME, nb_steps + nb_pre_steps), overwrite=True)

class EpisodeAccumulator(rl.callbacks.Callback):
    def __init__(self):
        self.reward_sum = 0
        self.episode_count = 0

    def on_episode_end(self, episode, logs={}):
        self.reward_sum += logs['episode_reward']
        self.episode_count += 1

    def reward_average(self):
        return self.reward_sum / self.episode_count
accumulator = EpisodeAccumulator()

dqn.test(env, nb_episodes=3, visualize=True, callbacks=[accumulator])
print("total reward: " + str(accumulator.reward_sum))


