from stable_baselines3 import PPO
from ev_env import EVChargingEnv

env = EVChargingEnv()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=50000)

model.save("ev_ppo_model")