from gym.envs.registration import register

register(
    id='continuousRLSeeker-v0',
    entry_point='continuousrlSeeker.envs:continuousRLSeeker',
)
