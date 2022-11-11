from utils.search import get_candidate
import json
search_space = {
    'graph': ['spatial', 'dis2', 'dis4', 'dis4+2'],
    'input_width': [6, 8, 10, 12],
    'num_input_modules': [1, 2, 3],
    'initial_block_type': ['basic', 'bottleneck', 'mbconv'],
    'initial_residual': ['null', 'block', 'module', 'dense'],
    'input_temporal_scales': [1, 2, 3, 'linear'],
    'initial_main_width': [6, 8, 10, 12],
    'num_main_levels': [1, 2],
    'num_main_level_modules': [1, 2, 3],
    'block_type': ['basic', 'bottleneck', 'mbconv'],
    'bottleneck_factor': [2, 4],
    'residual': ['null', 'block', 'module', 'dense'],
    'main_temporal_scales': [1, 2, 3, 'linear'],
    'temporal_kernel_size': [3, 5, 7, 9],
    'se': ['null', 'inner', 'outer', 'both'],
    'se_ratio': [2, 4],
    'se_type': ['relative', 'absolute'],
    'nonlinearity': ['relu', 'swish'],
    'attention': ['null', 'channel', 'frame', 'joint'],
    'pool': ['global', 'spatial']
}

candidate_history = []

PATH = "zero_cost_experiments/results.json"

for choice in search_space.keys():
    alternatives = search_space[choice]
    num_alternatives = len(alternatives)
    search_space[choice] = {}
    for alternative in alternatives:
        search_space[choice][alternative] = 1.0 / num_alternatives

with open(PATH) as file:
    results = json.load(file)


for i in range(20):
    candidate, candidate_string, candidate_history = get_candidate(search_space, candidate_history)
    # print(candidate)
    #print(candidate_string)
    if str(candidate) not in results:
        print("Candidate not in results shit is fucked")
        results[str(candidate)] = {}

    with open(PATH, "w") as f:
        json.dump(results, f)



    