# Importing the required modules
import optuna
import jsbgym
import gymnasium as gym
from stable_baselines3 import PPO

# Defining the objective function
def objective(trial):
  # Creating the environment
  env = gym.make("PA28-HeadingControlTask-Shaping.STANDARD-NoFG-v0", render_mode="flightgear")
  # Sampling the hyperparameters
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
  clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
  ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2)
  batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
  n_epochs = trial.suggest_int("n_epochs", 4, 16)
  # Creating the agent with tensorboard functionality
  model = PPO("MlpPolicy", env, learning_rate=learning_rate, clip_range=clip_range, ent_coef=ent_coef, batch_size=batch_size, n_epochs=n_epochs, tensorboard_log="./Optuna_Tensorboard/")
  # Training the agent
  model.learn(1000000) # or any other number of timesteps
  # Evaluating the agent
  rewards = []
  obs, info = env.reset()
  terminated = False
  while not terminated:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
  return sum(rewards) # or any other metric

# Creating the study with study name and storage for optuna-dashboard
study = optuna.create_study(direction="maximize", study_name="jsbgym", storage="sqlite:///db.sqlite3", pruner=optuna.pruners.MedianPruner()) # or "minimize"
# Running the optimization
study.optimize(objective, n_trials=50) # or any other number of trials
# Printing the best hyperparameters
print(study.best_params)
