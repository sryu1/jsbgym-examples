# Importing the required modules
import optuna
import jsbgym
import gymnasium as gym
from stable_baselines3 import PPO
import json

# Defining the objective function
def objective(trial):
    # Creating the environment
    env = gym.make("J3-HeadingControlTask-Shaping.STANDARD-NoFG-v0")
    # Sampling the hyperparameters
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    n_epochs = trial.suggest_int("n_epochs", 3, 30)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    learning_rate = trial.suggest_float("lr", 1e-5, 0.5, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)

    # Creating the agent with tensorboard functionality
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        batch_size=batch_size,
        n_epochs=n_epochs,
        tensorboard_log="./OTB/",
    )
    # Training the agent
    try:
        model.learn(10000000)  # or any other number of timesteps
        model.save(f"./Models/{trial.number}")
    except (AssertionError, ValueError) as e:
        print(e)
        raise optuna.exceptions.TrialPruned()
    # Evaluating the agent
    rewards = 0
    obs, info = env.reset(seed=42)
    terminated = False
    while not terminated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward
    return rewards


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
study.optimize(objective, n_trials=100)  # or any other number of trials

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
