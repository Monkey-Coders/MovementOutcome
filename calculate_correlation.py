from scipy import stats
import json

with open("zero_cost_experiments/results.json") as file:
    results = json.load(file)

auc = []
grad_norm = []
snip = []
synflow = []
fisher = []
grasp = []
for model in results:
    model_stats = results[model]
    auc.append(model_stats["best_auc"])
    grad_norm.append(model_stats["grad_norm"])
    snip.append(model_stats["snip"])
    synflow.append(model_stats["synflow"])
    fisher.append(model_stats["fisher"])
    grasp.append(model_stats["grasp"])

#print(auc)
#print(grad_norm)

spearman_rank_grad_norm = stats.spearmanr(auc, grad_norm)
spearman_rank_snip = stats.spearmanr(auc, snip)
spearman_rank_synflow = stats.spearmanr(auc, synflow)
spearman_rank_fisher = stats.spearmanr(auc, fisher)
spearman_rank_grasp = stats.spearmanr(auc, grasp)
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
print("Spearman rank for fisher")
print(spearman_rank_fisher)
print("="*80)
print("Spearman rank for grasp")
print(spearman_rank_grasp)
print("="*80)


result = {
    "grad_norm": {
        "correlation": spearman_rank_grad_norm.correlation,
        "pvalue": spearman_rank_grad_norm.pvalue
    },
    "snip": {
        "correlation": spearman_rank_snip.correlation,
        "pvalue": spearman_rank_snip.pvalue
    },
    "synflow": {
        "correlation": spearman_rank_synflow.correlation,
        "pvalue": spearman_rank_synflow.pvalue
    },
    "fisher": {
        "correlation": spearman_rank_fisher.correlation,
        "pvalue": spearman_rank_fisher
    },
    "grasp": {
        "correlation": spearman_rank_grasp.correlation,
        "pvalue": spearman_rank_grasp
    }
}
with open("zero_cost_experiments/spearman_rank.json", "w") as file:
    json.dump(result, file)
