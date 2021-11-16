import gym
import imageio

observations = []
for i in range(1000):
    env = gym.make('CarRacing-v0')
    env.reset()
    for s in range(1000):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action) # take a random action
        observations.append(obs)
        imageio.imwrite(f"out/carracing/obs_{i}_{s}.jpg",obs)
    env.close()
    