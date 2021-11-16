import gym
import imageio
import pickle

observations = []
for i in range(100):
    print(f"Iteration {i}/100...")
    env = gym.make('CarRacing-v0')
    env.reset()
    for s in range(100):
        action = env.action_space.sample()
        obs, _, _, _ = env.step(action) # take a random action
        observations.append(obs)
    env.close()

with open(r"out/carRacing.pickle", "wb") as output_file:
    pickle.dump(observations, output_file)
    