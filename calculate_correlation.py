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
flops = []
params = []
for model in results:
    try:
        model_stats = results[model]
        auc.append(model_stats["best_auc"])
        grad_norm.append(model_stats["grad_norm"])
        snip.append(model_stats["snip"])
        synflow.append(model_stats["synflow"])
        fisher.append(model_stats["fisher"])
        grasp.append(model_stats["grasp"])
        flops.append(model_stats["flops"])
        params.append(model_stats["params"])
    except KeyError:
        pass

#print(auc)
#print(grad_norm)

spearman_rank_grad_norm = abs(stats.spearmanr(auc, grad_norm, nan_policy="omit"))
spearman_rank_snip = abs(stats.spearmanr(auc, snip, nan_policy="omit"))
spearman_rank_synflow = abs(stats.spearmanr(auc, synflow, nan_policy="omit"))
spearman_rank_fisher = abs(stats.spearmanr(auc, fisher, nan_policy="omit"))
spearman_rank_grasp = abs(stats.spearmanr(auc, grasp, nan_policy="omit"))
spearman_rank_flops = abs(stats.spearmanr(auc, flops, nan_policy="omit"))
spearman_rank_params = abs(stats.spearmanr(auc, params, nan_policy="omit"))

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
print("Spearman rank for flops")
print(spearman_rank_flops)
print("="*80)
print("Spearman rank for params")
print(spearman_rank_params)
print("="*80)


result = {
    "grad_norm": {
        "correlation": round(spearman_rank_grad_norm.correlation,3),
        "pvalue": round(spearman_rank_grad_norm.pvalue,3)
    },
    "snip": {
        "correlation": round(spearman_rank_snip.correlation,3),
        "pvalue": round(spearman_rank_snip.pvalue,3)
    },
    "synflow": {
        "correlation": round(spearman_rank_synflow.correlation,3),
        "pvalue": round(spearman_rank_synflow.pvalue,3)
    },
    "fisher": {
        "correlation": round(spearman_rank_fisher.correlation,3),
        "pvalue": round(spearman_rank_fisher,3)
    },
    "grasp": {
        "correlation": round(spearman_rank_grasp.correlation,3),
        "pvalue": round(spearman_rank_grasp,3)
    },
    "flops": {
        "correlation": round(spearman_rank_flops.correlation,3),
        "pvalue": round(spearman_rank_flops,3)
    },
    "params": {
        "correlation": round(spearman_rank_params.correlation,3),
        "pvalue": round(spearman_rank_params,3)
    }
}
with open("zero_cost_experiments/spearman_rank.json", "w") as file:
    json.dump(result, file)
