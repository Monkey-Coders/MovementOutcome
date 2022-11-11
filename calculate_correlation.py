from scipy import stats
import json

with open("zero_cost_experiments/results.json") as file:
    results = json.load(file)

auc = []
grad_norm = []
snip = []
synflow = []
for model in results:
    model_stats = results[model]
    auc.append(model_stats["best_auc"])
    grad_norm.append(model_stats["grad_norm"])
    snip.append(model_stats["snip"])
    synflow.append(model_stats["synflow"])

#print(auc)
#print(grad_norm)

spearman_rank_grad_norm = stats.spearmanr(auc, grad_norm)
spearman_rank_snip = stats.spearmanr(auc, snip)
spearman_rank_synflow = stats.spearmanr(auc, synflow)
print("="*80)
print("Spearman rank for grad norm")
print(spearman_rank_grad_norm)
print("="*80)
print("Spearman rank for snip")
print(spearman_rank_snip)
print("="*80)
print("Spearman rank for synflow")
print(spearman_rank_synflow)
print("="*80)


