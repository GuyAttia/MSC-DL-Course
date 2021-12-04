import hw2.experiments as experiments

from hw2.experiments import load_experiment


# L = [2,4,8,16]
# K = [32]

# for l in L:
#     experiments.run_experiment('exp1_1', model_type='resnet', filters_per_layer=K, layers_per_block=l, pool_every=4)
    
L = [2,4,8,16]
K = [64]

for l in L:
    experiments.run_experiment('exp1_1', model_type='resnet', filters_per_layer=K, layers_per_block=l, pool_every=4)
    

# L = [2]
# K = [[32], 

# for k in K:
#     experiments.run_experiment('exp1_2', model_type='resnet', filters_per_layer=k, layers_per_block=L, pool_every=4)