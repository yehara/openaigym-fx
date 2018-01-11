from gym.envs.registration import register

register(
    id='FxEnv-v1',
    entry_point='fxenv.fxenv:FxEnv',
)
