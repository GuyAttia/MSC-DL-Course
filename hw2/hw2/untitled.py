import hw2.experiments as experiments

from hw2.experiments import load_experiment
import torch

#L = [2,4,8,16]
#K = [32]

#for l in L:
#    experiments.run_experiment('exp1_1', model_type='resnet', filters_per_layer=K, layers_per_block=l, pool_every= 4, hidden_dims=[100], batches= 50)
#    torch.cuda.empty_cache()
    
#L = [2,4,8,16]
#K = [64]

#for l in L:
#    experiments.run_experiment('exp1_1', model_type='resnet', filters_per_layer=K, layers_per_block=l, pool_every=4, hidden_dims=[100], batches= 50)
#    torch.cuda.empty_cache()
    

#L = 2
#K = [[32], [64], [128], [256]]

#for k in K:
#	experiments.run_experiment('exp1_2', model_type='resnet', filters_per_layer=k, layers_per_block=L, pool_every=4, hidden_dims=[100], batches= 500)
#	torch.cuda.empty_cache()
	
#L = 4
#K = [[32], [64], [128], [256]] 

#for k in K:
#	experiments.run_experiment('exp1_2', model_type='resnet', filters_per_layer=k, layers_per_block=L, pool_every=4, hidden_dims=[100], batches= 500)
#	torch.cuda.empty_cache()


#L = 8
#K = [[32], [64], [128], [256]] 

#for k in K:
#	experiments.run_experiment('exp1_2', model_type='resnet', filters_per_layer=k, layers_per_block=L, pool_every=4, hidden_dims=[100], batches= 500)
#	torch.cuda.empty_cache()


#L = [1,2,3,4]
#K = [64, 128, 256]

#for l in L:
#	experiments.run_experiment('exp1_3', model_type='resnet', filters_per_layer=K, layers_per_block=l, pool_every=4, hidden_dims=[100], batches= 500)
#	torch.cuda.empty_cache()
	
#L = [8,16,32]
#K = [32]

#for l in L:
#	experiments.run_experiment('exp1_4', model_type='resnet', filters_per_layer=K, layers_per_block=l, pool_every=4, hidden_dims=[100], batches= 500)
#	torch.cuda.empty_cache()


L = [8]
K = [64, 128, 256]

for l in L:
	experiments.run_experiment('exp1_4', model_type='resnet', filters_per_layer=K, layers_per_block=l, pool_every=2, hidden_dims=[100], batches= 500)
	torch.cuda.empty_cache()

