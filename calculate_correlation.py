from scipy import stats
import json
import matplotlib.pyplot as plt
import numpy as np

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
jacobian = []

auc_time = []
grad_norm_time = []
snip_time = []
synflow_time = []
fisher_time = []
grasp_time = []
flops_time = []
params_time = []
jacobian_time = []

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
        jacobian.append(model_stats["jacobian"])
        auc_time.append(model_stats["training_validation_time_in_seconds"])
        grad_norm_time.append(model_stats["grad_norm_time_in_seconds"])
        snip_time.append(model_stats["snip_time_in_seconds"])
        synflow_time.append(model_stats["synflow_time_in_seconds"])
        fisher_time.append(model_stats["fisher_time_in_seconds"])
        grasp_time.append(model_stats["grasp_time_in_seconds"])
        flops_time.append(model_stats["flops_time_in_seconds"])
        params_time.append(model_stats["params_time_in_seconds"])
        jacobian_time.append(model_stats["jacobian_time_in_seconds"])
    except KeyError:
        pass

spearman_rank_grad_norm = stats.spearmanr(auc, grad_norm)
spearman_rank_snip = stats.spearmanr(auc, snip)
spearman_rank_synflow = stats.spearmanr(auc, synflow)
spearman_rank_fisher = stats.spearmanr(auc, fisher)
spearman_rank_grasp = stats.spearmanr(auc, grasp)
spearman_rank_flops = stats.spearmanr(auc, flops)
spearman_rank_params = stats.spearmanr(auc, params)
spearman_rank_jacobian = stats.spearmanr(auc, jacobian)

results_list = [spearman_rank_grad_norm, spearman_rank_snip, spearman_rank_synflow, spearman_rank_fisher, spearman_rank_grasp, spearman_rank_flops, spearman_rank_params, spearman_rank_jacobian]

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
        "pvalue": round(spearman_rank_fisher.pvalue,3)
    },
    "grasp": {
        "correlation": round(spearman_rank_grasp.correlation,3),
        "pvalue": round(spearman_rank_grasp.pvalue,3)
    },
    "flops": {
        "correlation": round(spearman_rank_flops.correlation,3),
        "pvalue": round(spearman_rank_flops.pvalue,3)
    },
    "params": {
        "correlation": round(spearman_rank_params.correlation,3),
        "pvalue": round(spearman_rank_params.pvalue,3)
    },
    "jacobian": {
        "correlation": round(spearman_rank_jacobian.correlation,3),
        "pvalue": round(spearman_rank_jacobian.pvalue,3)
    }
}
with open("zero_cost_experiments/spearman_rank.json", "w") as file:
    json.dump(result, file)

# Calculate average time taken for each method
average_time = {
    "training": round(np.mean(auc_time), 3),
    "grad_norm": round(np.mean(grad_norm_time), 3),
    "snip": round(np.mean(snip_time), 3),
    "synflow": round(np.mean(synflow_time), 3),
    "fisher": round(np.mean(fisher_time), 3),
    "grasp": round(np.mean(grasp_time), 3),
    "flops": round(np.mean(flops_time), 3),
    "params": round(np.mean(params_time), 3),
    "jacobian": round(np.mean(jacobian_time), 3)
}

print(average_time)

#Plot average time with matplotlib with scatter
""" plt.scatter(average_time.keys(), average_time.values())
plt.xlabel("Method")
plt.ylabel("Average time taken (seconds)")
plt.title("Average time taken for each method")
# Save plt
plt.savefig("zero_cost_experiments/average_time.png") """

""" positive_results = {k: abs(v["correlation"]) for k, v in result.items()}
print(positive_results)
plt.scatter(positive_results.keys(), [x for x in positive_results.values()])
plt.xlabel("Method")
plt.ylabel("Spearman rank correlation")
plt.title("Spearman rank correlation for each method")
# Save plt
plt.show()
plt.savefig("zero_cost_experiments/spearman_rank_pos.png") """





