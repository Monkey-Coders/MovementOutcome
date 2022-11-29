import json

with open('zero_cost_experiments/results.json') as f:
    results = json.load(f)
    count = len([x for x in results.values() if x])
    print(F"We have trained a total of {count} models")
