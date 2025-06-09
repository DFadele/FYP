from stable_baselines3 import SAC
import gymnasium

env = gymnasium.make('Pendulum-v1')
model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("sac_pendulum")