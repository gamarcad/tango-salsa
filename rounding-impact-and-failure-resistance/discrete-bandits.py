from random import random, seed, randint
from utils import BernoulliArm
from math import sqrt, pow, log, e
from numpy.random import beta, default_rng
import matplotlib.pyplot as plt 


# the input data composed of the arm probabilities and the budget N 
prob =  [0.3404029692470838, 0.05408271474019088, 0.036055143160127257, 0.12937433722163308, 0.041357370095440084, 0.015906680805938492, 0.2788971367974549, 0.16436903499469777, 0.22375397667020147, 0.06256627783669141, 0.17709437963944857, 0.24602332979851538, 0.09968186638388123, 0.14103923647932132, 0.19936373276776245, 0.016967126193001062, 0.043478260869565216, 0.003181336161187699, 0.053022269353128315, 0.042417815482502653, 0.015906680805938492, 0.24602332979851538, 0.15482502651113467, 0.09544008483563096, 0.1633085896076352, 0.031813361611876985, 0.02332979851537646, 0.21208907741251326, 0.021208907741251327, 0.031813361611876985, 0.09862142099681867, 0.05620360551431601, 0.053022269353128315, 0.0021208907741251328, 0.0021208907741251328, 0.0010604453870625664, 0.0010604453870625664, 0.031813361611876985, 0.04029692470837752, 0.02014846235418876, 0.013785790031813362, 0.10180275715800637, 0.010604453870625663, 0.04029692470837752, 0.061505832449628844, 0.016967126193001062, 0.08377518557794274, 0.10180275715800637, 0.03499469777306469, 0.5312831389183457, 0.04772004241781548, 0.05938494167550371, 0.05408271474019088, 0.042417815482502653, 0.10286320254506894, 0.3117709437963945, 0.031813361611876985, 0.10922587486744433, 0.06998939554612937, 0.04878048780487805, 0.043478260869565216, 0.04559915164369035, 0.027571580063626724, 0.2704135737009544, 0.07317073170731707, 0.09013785790031813, 0.04559915164369035, 0.07211028632025451, 0.23860021208907742, 0.16861081654294804, 0.1420996818663839, 0.05514316012725345, 0.06468716861081654, 0.003181336161187699, 0.003181336161187699, 0.02863202545068929, 0.06468716861081654, 0.006362672322375398, 0.2799575821845175, 0.016967126193001062, 0.06468716861081654, 0.17179215270413573, 0.1474019088016967, 0.006362672322375398, 0.019088016967126194, 0.11983032873807, 0.10286320254506894, 0.13361611876988336, 0.23223753976670203, 0.03711558854718982, 0.08695652173913043, 0.06786850477200425, 0.0816542948038176, 0.051961823966065745, 0.15270413573700956, 0.23966065747614, 0.17709437963944857, 0.3647932131495228, 0.10604453870625663, 0.4305408271474019]
epsilon = 0.1
N = 1000
K = len(prob)

def argmax(lst):
    if not lst:
        raise ValueError("La liste ne doit pas être vide.")
    
    max_value = lst[0]
    max_index = 0
    
    for i in range(1, len(lst)):
        if lst[i] > max_value:
            max_value = lst[i]
            max_index = i
    
    return max_index


def mean(l): return sum(l) / len(l)

def approximate( score  ) -> int:
    return int( score * 10**10)

def UCB_score( s_i, n_i, t, i ) -> float:
    exploitation_term = s_i / n_i
    exploration_term = sqrt((2 * log(t, e)) / n_i)
    return exploitation_term + exploration_term

def Thompson_score( s_i, n_i, t, i ) -> float:
    rng = default_rng(10000 * t + i)
    return rng.beta( s_i + 1, n_i - s_i + 1 )



def EGreedy_score( s_i, n_i, t, i ) -> float:
    seed(10000 * t + i)
    explore = random() < epsilon   
    if explore:
        return randint(1, 10 * K)
    else:
        return s_i / n_i
        






# evaluate the discrete multi-armed bandits
def MultiArmedBandits( score_function, budget = N, iteration = 200):
    rewards_over_time = {}

    for arms_seed in range(iteration):
        print(f"\t - Seed {arms_seed}")
        pulls = [ 1 for  _ in range(K) ]
        arms = [ BernoulliArm( p, seed = arms_seed ) for p in prob ]
        local_rewards = [ arms[i].pull(t = 1) for i in range(K)]
        
    
        for t in range(1, budget + 1):
            
            scores = []
            for i in range(K):
                scores.append(
                    score_function( s_i = local_rewards[i], n_i = pulls[i], t = t, i = i )
                )
     
        
            M = argmax(scores)
            assert M is not None
            pulls[M] += 1
            local_rewards[M] += arms[M].pull(t = t)

            if t not in rewards_over_time:
                rewards_over_time[t] = []

            rewards_over_time[t].append(sum(local_rewards))

    return (
        rewards_over_time.keys(),
        {t: mean(rewards) for (t,rewards) in rewards_over_time.items()}, 
    )

# evaluate the discrete multi-armed bandits
def DiscreteMultiArmedBandits( score_function, budget = N, iteration = 200):
    rewards_over_time = {}

    for arms_seed in range(iteration):
        print(f"\t - Seed {arms_seed}")
        pulls = [ 1 for  _ in range(K) ]
        arms = [ BernoulliArm( p, seed = arms_seed ) for p in prob ]
        local_rewards = [ arms[i].pull(t = 1) for i in range(K)]
        
    
        for t in range(1, budget + 1):
            
            scores = []
            for i in range(K):
                scores.append(
                    approximate( score_function( s_i = local_rewards[i], n_i = pulls[i], t = t , i = i) )
                )
     
        
            M = argmax(scores)
            assert M is not None
            pulls[M] += 1
            local_rewards[M] += arms[M].pull(t = t)

            if t not in rewards_over_time:
                rewards_over_time[t] = []

            rewards_over_time[t].append(sum(local_rewards))
    
    return (
        rewards_over_time.keys(),
        {t: mean(rewards) for (t,rewards) in rewards_over_time.items()}, 
    )




if __name__ == "__main__":
  
    # evaluate e-greedy
    print("Running EGreedy...")
    score_function = EGreedy_score
    times, egreedy_rot = MultiArmedBandits( score_function=score_function )
    _, egreedy_rot_dis = DiscreteMultiArmedBandits( score_function=score_function )

    # evaluate UCB
    print("Running UCB")
    score_function = UCB_score
    _, ucb_rot = MultiArmedBandits( score_function=score_function )
    _, ucb_rot_dis = DiscreteMultiArmedBandits( score_function=score_function )

    # evaluate thompson sampling
    print("Running Thompson...")
    score_function = Thompson_score
    _, thompson_rot = MultiArmedBandits( score_function=score_function )
    _, thompson_rot_dis = DiscreteMultiArmedBandits( score_function=score_function )

    # export the results
    with open("difference-rewards-standard-discrete-on-movie-lens.csv", 'w') as file:
        file.write("time,ucb,ucb-discrete,epsilon,epsilon-discrete,thompson,thompson-discrete\n")

        for t in times:
            file.write(f"{t},{ucb_rot[t]},{ucb_rot_dis[t]},{egreedy_rot[t]},{egreedy_rot_dis[t]},{thompson_rot[t]},{thompson_rot_dis[t]}\n")

