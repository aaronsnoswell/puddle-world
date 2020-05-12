from gym.envs.registration import register

from puddle_world.envs.puddle_world_env import PuddleWorldEnv, CanonicalPuddleWorldEnv


register(
    id="PuddleWorld-v0", entry_point="plane_world.envs:PuddleWorldEnv",
)
