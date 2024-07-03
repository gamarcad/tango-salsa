from random import random, seed, randint, choices
from utils import BernoulliArm, argmax
from math import sqrt, pow, log
from numpy.random import beta, default_rng
import matplotlib.pyplot as plt 

# the input data composed of the arm probabilities and the budget N 
prob =  [0.3404029692470838, 0.05408271474019088, 0.036055143160127257, 0.12937433722163308, 0.041357370095440084, 0.015906680805938492, 0.2788971367974549, 0.16436903499469777, 0.22375397667020147, 0.06256627783669141, 0.17709437963944857, 0.24602332979851538, 0.09968186638388123, 0.14103923647932132, 0.19936373276776245, 0.016967126193001062, 0.043478260869565216, 0.003181336161187699, 0.053022269353128315, 0.042417815482502653, 0.015906680805938492, 0.24602332979851538, 0.15482502651113467, 0.09544008483563096, 0.1633085896076352, 0.031813361611876985, 0.02332979851537646, 0.21208907741251326, 0.021208907741251327, 0.031813361611876985, 0.09862142099681867, 0.05620360551431601, 0.053022269353128315, 0.0021208907741251328, 0.0021208907741251328, 0.0010604453870625664, 0.0010604453870625664, 0.031813361611876985, 0.04029692470837752, 0.02014846235418876, 0.013785790031813362, 0.10180275715800637, 0.010604453870625663, 0.04029692470837752, 0.061505832449628844, 0.016967126193001062, 0.08377518557794274, 0.10180275715800637, 0.03499469777306469, 0.5312831389183457, 0.04772004241781548, 0.05938494167550371, 0.05408271474019088, 0.042417815482502653, 0.10286320254506894, 0.3117709437963945, 0.031813361611876985, 0.10922587486744433, 0.06998939554612937, 0.04878048780487805, 0.043478260869565216, 0.04559915164369035, 0.027571580063626724, 0.2704135737009544, 0.07317073170731707, 0.09013785790031813, 0.04559915164369035, 0.07211028632025451, 0.23860021208907742, 0.16861081654294804, 0.1420996818663839, 0.05514316012725345, 0.06468716861081654, 0.003181336161187699, 0.003181336161187699, 0.02863202545068929, 0.06468716861081654, 0.006362672322375398, 0.2799575821845175, 0.016967126193001062, 0.06468716861081654, 0.17179215270413573, 0.1474019088016967, 0.006362672322375398, 0.019088016967126194, 0.11983032873807, 0.10286320254506894, 0.13361611876988336, 0.23223753976670203, 0.03711558854718982, 0.08695652173913043, 0.06786850477200425, 0.0816542948038176, 0.051961823966065745, 0.15270413573700956, 0.23966065747614, 0.17709437963944857, 0.3647932131495228, 0.10604453870625663, 0.4305408271474019]
epsilon = 0.1
N = 1000
K = len(prob)






def approximate( score  ) -> int:
    return int( score * 10**10)

def UCB_score( s_i, n_i, t ) -> float:
    return (s_i / n_i) + sqrt( (2 * log(t))  / n_i )

def Thompson_score( s_i, n_i, t ) -> float:
    rng = default_rng(2022 + t)
    return rng.beta( s_i + 1, n_i - s_i + 1 )

epsilon_memory = {}
def EGreedy_score( s_i, n_i, t ) -> float:
    global epsilon_memory
    if t not in epsilon_memory:
        explore = random() < epsilon
        epsilon_memory[t] = explore
    
    if epsilon_memory[t]:
        seed(t)
        return randint(1, K)
    else:
        return s_i / n_i
        
def mean(l): return sum(l) / len(l)


# evaluate the discrete multi-armed bandits
def DiscreteMultiArmedBandits( score_function, drop_factor, budget = N, offline_time = N // 2):
    rewards_over_time_without_saving = {}
    rewards_over_time_with_saving = {}  

    for arms_seed in range(200):
        pulls = [ 1 for  _ in range(K) ]
        arms = [ BernoulliArm( p, seed = arms_seed ) for p in sorted(prob, reverse=True) ]
        local_rewards = [ arms[i].pull(t = 1) for i in range(K)]
        rewards_register = [ 0 for  _ in range(K) ]
        

        # the drop arms will be highest ones (see sorted(prob, reverse=True))
        offline_arms = [ i for i in range(int(K * drop_factor)) ]
        
        for t in range(1, budget + 1):
            # at the offline time, all offline data owners are deleting their local storage, modelled here as the deletion of their local rewards
            if t == offline_time:
                for i in offline_arms:
                    local_rewards[i] = 0


            # online data owners are updating their local score
            for i in range(K):
                if t < offline_time or i not in offline_arms:
                    rewards_register[i] = local_rewards[i]

            
            scores = []
            for i in range(K):
                if t < offline_time or i not in offline_arms :
                    if offline_time <= t and drop_factor == 1: raise Exception("WHat ?" + str(t) + str(i)) 
                    scores.append(
                        approximate( score_function( s_i = local_rewards[i], n_i = pulls[i], t = t ) )
                    )
                else:
                    scores.append(None)
        
            
            if not all([ score is None for score in scores ]):
                M = argmax(scores)
                assert M is not None
                pulls[M] += 1
                local_rewards[M] += arms[M].pull(t = t)
            else:
                assert drop_factor == 1

            

            if t not in rewards_over_time_without_saving:
                rewards_over_time_without_saving[t] = []
                rewards_over_time_with_saving[t] = []

            rewards_over_time_without_saving[t].append(sum(local_rewards))
            rewards_over_time_with_saving[t].append(sum(rewards_register))
            

           
        
    
    
    return (
        rewards_over_time_without_saving.keys(),
        {t: mean(rewards) for (t,rewards) in rewards_over_time_without_saving.items()}, 
        {t: mean(rewards) for (t,rewards) in rewards_over_time_with_saving.items()}
    )


print("Running 0...")
times, without_saving_0, with_saving_0 = DiscreteMultiArmedBandits(UCB_score, drop_factor=0)

print("Running 25...")
times, without_saving_25, with_saving_25 = DiscreteMultiArmedBandits(UCB_score, drop_factor=.25)

print("Running 50...")
times, without_saving_50, with_saving_50 = DiscreteMultiArmedBandits(UCB_score, drop_factor=.5)

print("Running 75...")
times, without_saving_75, with_saving_75 = DiscreteMultiArmedBandits(UCB_score, drop_factor=.75)

print("Running 100...")
times, without_saving_100, with_saving_100 = DiscreteMultiArmedBandits(UCB_score, drop_factor=1)


# compute and export the cumulative rewards 
with open("failures-resistance.csv", 'w') as file:
    # header.
    file.write("time,0-without-saving,0-with-saving,25-without-saving,25-with-saving,50-without-saving,50-with-saving,75-without-saving,75-with-saving,100-without-saving,100-with-saving\n")
    
    for t in times:
        file.write(f"{t},{without_saving_0[t]},{with_saving_0[t]},{without_saving_25[t]},{with_saving_25[t]},{without_saving_50[t]},{with_saving_50[t]},{without_saving_75[t]},{with_saving_75[t]}, {without_saving_100[t]},{with_saving_100[t]}\n")
    
   

    

