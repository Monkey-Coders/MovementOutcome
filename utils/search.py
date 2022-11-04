# https://github.com/daniegr/KBestSearch


import random
import math


def print_status(candidate_string, performance, candidate_num, best, best_performance, best_candidate_num, population, temperature=None, unsuccessful_trials=None):
    print('###############################################')
    
    # Results of current candidate
    print('Candidate {0} {1} had performance {2:.5f}.'.format(candidate_num, candidate_string, performance))
    
    # Current best candidate
    print('\nThe best trial is candidate {0} {1} with performance {2:.5f}.'.format(best_candidate_num, best, best_performance))
    
    # Current population
    if len(population) > 0:
        print('\nThe population is:')
        for n in range(1, len(population)+1):
            candidate_string, performance, candidate_num = population[-n]
            print(' {0}. Candidate {1} {2} with performance {3:.5f}.'.format(n, candidate_num, candidate_string, performance))
            
    # Current temperature and number of unsuccessful trials at this temperature
    if not temperature is None:
        print('\nThe temperature is {0} with {1} unsuccessful trials'.format(temperature, unsuccessful_trials))
   
    print('###############################################\n\n')
    
    
def update_best(candidate_string, performance, candidate_num, best, best_performance, best_candidate_num):
    
    # Compare to best candidate
    if performance > best_performance:
        best_performance = performance
        best_candidate_num = candidate_num
        best = candidate_string
    
    return best, best_performance, best_candidate_num


def get_candidate(search_space, candidate_history):
    
    # Define max attempts as twice the number of unique candidates
    max_attempts = 2
    for choice in search_space.keys():
        max_attempts *= len(list(search_space[choice].keys()))
    
    # Fetch candidate
    attempt = 1
    while max_attempts >= attempt:
    
        # Initialize candidate
        candidate = {}
        candidate_string = ''

        # Iterate over choices
        for choice in search_space.keys():

            # Select alternatives based on probability in search space
            random_number = random.uniform(0, 1)
            alternatives = search_space[choice].keys()
            value = 0.0
            for alternative in alternatives:
                alternative_prob = search_space[choice][alternative]
                value += alternative_prob
                if value >= random_number:
                    candidate[choice] = alternative
                    candidate_string += '|{0}-{1}'.format(choice, alternative) if len(candidate_string) > 1 else '{0}-{1}'.format(choice, alternative)
                    break

        if candidate_string not in candidate_history:
            candidate_history.append(candidate_string)
            break
        else:
            attempt += 1
            if attempt > max_attempts:
                candidate = None
                candidate_string = ''
    
    return candidate, candidate_string, candidate_history  


def update_search_space(population, temperature):
    # Determine contribution of each alternative in search space
    search_space_count = {}
    for n, (candidate_string, _, _) in enumerate(population, 1):
        structure = candidate_string.split('|')
        for pair in structure:
            choice, alternative = pair.split('-')
            if not choice in search_space_count.keys():
                search_space_count[choice] = {}
            try:
                search_space_count[choice][alternative] += n
            except:
                search_space_count[choice][alternative] = n
                                    
    # Compute search space probabilities based on softmax
    search_space = {}
    for choice in search_space_count.keys():
        exp_sum = 0.0
        for alternative in search_space_count[choice].keys():
            exp_sum += math.exp(search_space_count[choice][alternative]/temperature)
        search_space[choice] = {}
        for alternative in search_space_count[choice].keys():
            search_space[choice][alternative] = math.exp(search_space_count[choice][alternative]/temperature) / exp_sum
                
    return search_space