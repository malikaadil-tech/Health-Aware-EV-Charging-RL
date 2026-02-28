import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ev_env import EVChargingEnv

env = EVChargingEnv()
model = PPO.load("ev_ppo_model")

obs, _ = env.reset()

soc_list = []
soh_list = []
price_list = []
action_list = []

done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    soc_list.append(env.soc)
    soh_list.append(env.soh)
    price_list.append(env.price_profile[env.step_count-1])
    action_list.append(action[0])

plt.figure()
plt.plot(soc_list)
plt.title("SOC over Time")
plt.show()

plt.figure()
plt.plot(soh_list)
plt.title("SOH Degradation")
plt.show()

plt.figure()
plt.plot(price_list, label="Price")
plt.plot(action_list, label="Action")
plt.legend()
plt.title("Price vs Charging Action")
plt.show()