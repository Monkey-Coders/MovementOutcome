from scipy import stats
import json

with open("zero_cost_experiments/results.json") as file:
    results = json.load(file)

auc = []
grad_norm = []
num_flops = []
for model in results:
    model_stats = results[model]
    auc.append(model_stats["val_accuracy"])
    grad_norm.append(model_stats["grad_norm"])
    num_flops.append(model_stats["num_flops"])

#print(auc)
#print(grad_norm)

spearman_rank_grad_norm = stats.spearmanr(auc, grad_norm)
spearman_rank_num_flops = stats.spearmanr(auc, num_flops)
print("="*80)
print("Spearman rank for grad norm")
print(spearman_rank_grad_norm)
print("="*80)
print("Spearman rank for num flops")
print(spearman_rank_num_flops)
print("="*80)


