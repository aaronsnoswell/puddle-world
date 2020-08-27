from gym.envs.registration import register

from puddle_world.envs.puddle_world_env import (
    ExplicitPuddleWorldEnv,
    CanonicalPuddleWorldEnv,
)

register(
    id="ExplicitPuddleWorld-v0", entry_point="puddle_world.envs:ExplicitPuddleWorldEnv",
)

register(
    id="CanonicalPuddleWorld-v0",
    entry_point="puddle_world.envs:CanonicalPuddleWorldEnv",
)
