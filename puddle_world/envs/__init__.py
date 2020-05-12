from gym.envs.registration import register

register(
    id="PuddleWorld-v0", entry_point="plane_world.envs:PuddleWorldEnv",
)
