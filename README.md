# README

## PPO Training for Johnson Subgraph Environment

This implementation of PPO is based on the [CleanRL repository](https://github.com/vwxyzjn/cleanrl).

This repository contains `ppo.py`, which implements Proximal Policy Optimization (PPO) for training an agent in the Johnson subgraph environment. You can run the script with various arguments to customize the training process.

### Running the Script
To start training, use the following command:
```sh
python3 ppo.py --env_id="JohnsonSubgraph-v0;n=10;k=5;action_is_subset=True;max_length=20;init_max_length=0"
```

### Configuring the Environment
The `--env_id` argument configures the environment and should contain all parameters separated by `;`. The general format is:
```
JohnsonSubgraph-v0;n=<value>;k=<value>;action_is_subset=<True/False>;max_length=<value>;init_max_length=<value>
```

- `n`: The size of the Johnson graph
- `k`: The subset size
- `action_is_subset`: If `True`, actions add entire subsets; if `False`, actions add elements to subsets
- `max_length`: Maximum length of training episodes
- `init_max_length`: Initial number of subsets in the environment

You can modify these parameters to customize the environment setup for training.

### Example Configurations
#### Training with Element-wise Subset Addition
```sh
python3 ppo.py --env_id="JohnsonSubgraph-v0;n=8;k=3;action_is_subset=False;max_length=15;init_max_length=2"
```

#### Training with Full Subset Addition
```sh
python3 ppo.py --env_id="JohnsonSubgraph-v0;n=12;k=6;action_is_subset=True;max_length=25;init_max_length=5"
```

### Using M2 Backend
To use the M2 backend for linearity checks, you need to have `Macaulay2` and `SageMath` installed on your system. Ensure they are properly configured before running the script.

### Notes

### Tracking Training Progress
You can check out the latest Weights & Biases (wandb) runs [here](https://wandb.ai/kibrq/Monomial%20Ideals?nw=nwuserkibrq).

- Ensure that all arguments in `--env_id` are separated by `;` and match the expected key-value format.
- The PPO training script supports additional hyperparameter tuning via command-line arguments.
- Use `--help` to see all available options:
  ```sh
  python3 ppo.py --help
  ```
