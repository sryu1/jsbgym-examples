# Importing the required modules
import optuna
import jsbgym
import gymnasium as gym
from stable_baselines3 import PPO
import json

# Defining the objective function
def objective(trial):
    # Creating the environment
    env = gym.make("PA28-HeadingControlTask-Shaping.STANDARD-NoFG-v0")
    # Sampling the hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 3e-3)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    n_epochs = trial.suggest_int("n_epochs", 3, 30)
    # Creating the agent with tensorboard functionality
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        clip_range=clip_range,
        ent_coef=ent_coef,
        batch_size=batch_size,
        n_epochs=n_epochs,
        tensorboard_log="./Optuna_Tensorboard/",
    )
    # Training the agent
    model.learn(3500000)  # or any other number of timesteps
    # Evaluating the agent
    rewards = []
    obs, info = env.reset(seed=42)
    terminated = False
    while not terminated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
    return sum(rewards)


# Creating a dictionary to store the trial data
trial_data = {}

# Creating the study with study name and storage for optuna-dashboard
study = optuna.create_study(
    direction="maximize",
    study_name="jsbgym",
    storage="sqlite:///db.sqlite3",
    pruner=optuna.pruners.MedianPruner(),
)  # or "minimize"
# Running the optimization
study.optimize(objective, n_trials=50)  # or any other number of trials

# Looping through the trials and adding them to the dictionary
for trial in study.trials:
    trial_data[trial.number] = trial.value

# Printing the best hyperparameters
print(study.best_params)

# Writing the trial data and hyperparameters to a JSON file
with open("trial_data.json", "w") as datafile:
    for trial in study.trials:
        trial_data[trial.number] = {"value": trial.value, "params": trial.params}
    json.dump(trial_data, datafile)
