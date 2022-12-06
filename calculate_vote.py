import json
from itertools import combinations

def init(names):
    PATH = "zero_cost_experiments/results.json"
    with open(PATH) as file:
        results = json.load(file)
        
    metrics = {}
    auc = []

    for key in results:
        value = results[key]
        for name in names:
            if name not in metrics:
                metrics[name] = []
            metrics[name].append(value[name])
            
        auc.append(value["best_auc"])
    return (auc, metrics)

def vote(mets, gt):
    numpos = 0
    for m in mets:
        numpos += 1 if m > 0 else 0
    if numpos >= len(mets)/2:
        sign = +1
    else:
        sign = -1
    return sign*gt


def calc(auc, metrics, comb):
    num_pts = len(auc)
    tot=0
    right=0
    for i in range(num_pts):
        for j in range(num_pts):
            if i!=j:
                diff = auc[i] - auc[j]
                if diff == 0:
                    continue
                diffsyn = []
                for m in comb:
                    diffsyn.append(metrics[m][i] - metrics[m][j])
                same_sign = vote(diffsyn, diff)
                right += 1 if same_sign > 0 else 0
                tot += 1
    votes = right/tot
    return (comb, votes)
    
def get_all_combinations(names):
    list_combinations = list()
    for n in range(2, len(names) + 1):
        list_combinations += list(combinations(names, n))
    return list_combinations

if __name__ == "__main__":
    names = ["synflow", "snip", "jacobian", "flops", "params", "fisher", "grasp", "grad_norm" ]
    auc, metrics = init(names)
        
    comb = get_all_combinations(names)
    
    D = {}
    for c in comb:
        a, votes = calc(auc, metrics, c)
        D[str(a)] = votes
    D = dict(sorted(D.items(), key=lambda item: item[1], reverse=True))

    with open("zero_cost_experiments/vote_combinations.json", "w") as file:
        json.dump(D, file)