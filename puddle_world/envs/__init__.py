
from gym.envs.registration import register

register(
    id='PuddleWorldl-v0',
    entry_point='plane_world.envs:PuddleWorldEnv',
)
