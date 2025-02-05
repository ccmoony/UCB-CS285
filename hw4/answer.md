## Problem 1
```shell
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_0_iter.yaml
```
![inital_loss](data/cheetah-cs285-v0_cheetah_0iter_l1_h32_mpcrandom_horizon10_actionseq1000_04-02-2025_14-03-38/itr_0_loss_curve.png)

Change the hidden size to 64 and run again:

![hidden_size_64](data/cheetah-cs285-v0_cheetah_0iter_l1_h64_mpcrandom_horizon10_actionseq1000_04-02-2025_14-06-01/itr_0_loss_curve.png)

Change the num_layers to 4 and run again:

![num_layers_4](data/cheetah-cs285-v0_cheetah_0iter_l4_h32_mpcrandom_horizon10_actionseq1000_04-02-2025_14-06-38/itr_0_loss_curve.png)
## Problem 2
```shell

python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_1_iter.yaml
```
![loss](data/obstacles-cs285-v0_obstacles_single_l2_h250_mpcrandom_horizon10_actionseq1000_04-02-2025_16-40-40/itr_0_loss_curve.png)

The eval_return value is -28.391082763671875.
## Problem 3
```shell
python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_multi_iter.yaml
```
![obstacles-eval_return](figures/obstacles_eval_return.png)
```shell
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_multi_iter.yaml
```
![reacher-eval_return](figures/reacher_eval_return.png)
```shell
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml
```
![halfcheetah-eval_return](figures/halfcheetah_eval_return.png)
## Problem 4
### Ablation study on ensemble_size
```shell
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_ablation.yaml
```
![ensemble_size_eval](figures/ensemble_eval.png)
![ensemble_size_loss](figures/ensemble_loss.png)
## Problem 5
```shell
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_cem.yaml
```
![halfcheetah-cem](figures/halfcheetah_cem.png)
### Problem 6
```shell
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_mbpo.yaml --sac_config_file experiments/sac/halfcheetah_clipq.yaml
```
![halfcheetah-mbpo](figures/halfcheetah_mbpo.png)

where purple represents rollout length of 10, yellow represents rollout length of 1 and blue represents rollout length of 0.
