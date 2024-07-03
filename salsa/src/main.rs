use rand::thread_rng;
use rand::{rngs::StdRng, Rng, SeedableRng};
use statrs::distribution::Beta;
use std::iter::zip;
use std::time::Instant;
use std::{collections::HashMap, ops::Div};

mod mpc;
mod salsa;

fn argmax(values: &Vec<&u64>) -> usize {
    values
        .iter()
        .enumerate()
        .max_by_key(|&(_, &&item)| item)
        .map(|(index, _)| index)
        .unwrap()
}

// derive the time step t into a seed
fn time_step_to_seed(t: u32) -> (u8, u8, u8, u8) {
    let bytes = t.to_le_bytes();
    (bytes[0], bytes[1], bytes[2], bytes[3])
}

fn update_seed_to_time_step(seed: [u8; 32], iteration: u32, t: u32) -> [u8; 32] {
    let (a, b, c, d) = time_step_to_seed(t);
    let (e, f, g, h) = time_step_to_seed(iteration);
    let mut updated_seed = seed.clone();
    updated_seed[1] = a;
    updated_seed[2] = b;
    updated_seed[3] = c;
    updated_seed[4] = d;
    updated_seed[5] = e;
    updated_seed[6] = f;
    updated_seed[7] = g;
    updated_seed[8] = h;
    updated_seed
}

fn update_seed_to_time_step_and_arm(seed: [u8; 32], iteration: u32, t: u32, i: u8) -> [u8; 32] {
    let (a, b, c, d) = time_step_to_seed(t);
    let (e, f, g, h) = time_step_to_seed(iteration);
    let mut updated_seed = seed.clone();
    updated_seed[0] = a;
    updated_seed[1] = b;
    updated_seed[2] = c;
    updated_seed[3] = d;
    updated_seed[4] = e;
    updated_seed[5] = f;
    updated_seed[6] = g;
    updated_seed[7] = h;
    for index in 8..31 {
        updated_seed[index] = i;
    }
    updated_seed
}

// create the (discrete) multi-armed bandits algorithm for UCB, e-greedy and Thompson Sampling
trait Strategy {
    fn score(&mut self, iteration: u32, i: u32, s_i: usize, n_i: usize, t: u32) -> f64;
}

struct UCB {}
impl Strategy for UCB {
    fn score(&mut self, _iteration: u32, i: u32, s_i: usize, n_i: usize, t: u32) -> f64 {
        let exploitation_term = s_i as f64 / n_i as f64;
        let exploration_term = f64::sqrt(2f64 * f64::ln(t as f64) / (n_i as f64));
        return exploitation_term + exploration_term;
    }
}

struct ThompsonSampling {
    seed: [u8; 32],
    memory: HashMap<(u32, u32, u32), f64>,
}

impl ThompsonSampling {
    fn new() -> Self {
        Self {
            seed: [0u8; 32],
            memory: HashMap::new(),
        }
    }
}

impl Strategy for ThompsonSampling {
    fn score(&mut self, iteration: u32, i: u32, s_i: usize, n_i: usize, t: u32) -> f64 {
        // compute the seed that must be chosen at time step
        //let seed_at_time_step = update_seed_to_time_step_and_arm(self.seed, iteration, t, i as u8);
        if self.memory.contains_key(&(iteration, t, i)) {
            return *self.memory.get(&(iteration, t, i)).unwrap();
        } else {
            let distribution = Beta::new((s_i + 1) as f64, (n_i - s_i + 1) as f64).unwrap();
            //let mut rng: StdRng = SeedableRng::from_seed(seed_at_time_step);
            let mut rng = thread_rng();
            let score: f64 = rng.sample(distribution);
            self.memory.insert((iteration, t, i), score);
            return score;
        }
    }
}

struct EpsilonGreedy {
    epsilon: f64,
    epsilon_seed: [u8; 32],
    random_score_seed: [u8; 32],
    explore: usize,
    exploit: usize,
    epsilon_memory: HashMap<(u32, u32), f64>,
    score_memory: HashMap<(u32, u32, u32), f64>,
}
impl EpsilonGreedy {
    fn new() -> Self {
        Self {
            epsilon: 0.1f64,
            epsilon_seed: [0u8; 32],
            random_score_seed: [0u8; 32],
            explore: 0,
            exploit: 0,
            epsilon_memory: HashMap::new(),
            score_memory: HashMap::new(),
        }
    }
}
impl Strategy for EpsilonGreedy {
    fn score(&mut self, iteration: u32, i: u32, s_i: usize, n_i: usize, t: u32) -> f64 {
        // compute the seed that must be chosen at time step
        let epsilon_seed_at_time_step = update_seed_to_time_step(self.epsilon_seed, iteration, t);
        //let mut rng: StdRng = SeedableRng::from_seed(epsilon_seed_at_time_step);
        let epsilon = {
            if self.epsilon_memory.contains_key(&(iteration, t)) {
                *self.epsilon_memory.get(&(iteration, t)).unwrap()
            } else {
                let mut rng = thread_rng();
                let epsilon: f64 = rng.gen();
                self.epsilon_memory.insert((iteration, t), epsilon.clone());
                epsilon
            }
        };

        if self.epsilon < epsilon {
            return s_i as f64 / n_i as f64;
        } else {
            let score = {
                if self.score_memory.contains_key(&(iteration, t, i)) {
                    *self.score_memory.get(&(iteration, t, i)).unwrap()
                } else {
                    let mut rng = thread_rng();
                    let score: f64 = rng.gen();
                    self.score_memory.insert((iteration, t, i), score.clone());
                    score
                }
            };
            return score;
        }
    }
}

#[derive(Clone)]
struct BernoulliArm {
    seed: [u8; 32],
    prob: f64,
    index: u8,
}

impl BernoulliArm {
    pub fn new(seed: [u8; 32], prob: f64, index: u8) -> Self {
        assert!(0f64 <= prob && prob <= 1f64);
        Self { seed, prob, index }
    }

    pub fn get_reward(&self, iteration: u32, t: u32) -> usize {
        let reward_seed_at_time_step =
            update_seed_to_time_step_and_arm(self.seed, iteration, t, self.index);
        let mut rng: StdRng = SeedableRng::from_seed(reward_seed_at_time_step);
        //let mut rng = thread_rng();
        let random_value: f64 = rng.gen();
        if random_value < self.prob {
            return 1;
        } else {
            return 0;
        }
    }
}

struct MultiArmedBanditsExecution {
    rewards: Vec<usize>,
    pulls: Vec<usize>,
    total_rewards: HashMap<u32, Vec<usize>>, // Associates at each time t the total cumulative rewards for all the iterations
    number_of_arms: usize,
    number_of_iterations: u32,
    arms: Vec<BernoulliArm>,
}

impl MultiArmedBanditsExecution {
    pub fn new(arms: Vec<BernoulliArm>, number_of_iterations: u32) -> Self {
        // create the list of rewards and pulls with the appropriate number of iterations
        let rewards = Vec::new();
        let pulls = Vec::new();

        return Self {
            rewards,
            pulls,
            total_rewards: HashMap::new(),
            number_of_arms: arms.len(),
            arms,
            number_of_iterations,
        };
    }

    pub fn number_of_arms(&self) -> usize {
        return self.number_of_arms;
    }

    pub fn start(&mut self, iteration: u32) {
        self.rewards.clear();
        self.pulls.clear();

        // pull each arm once
        for i in 0..self.number_of_arms {
            let arm: &BernoulliArm = self.arms.get(i).unwrap();
            let reward = arm.get_reward(iteration, 0);
            self.rewards.push(reward);
            self.pulls.push(1);
        }
    }

    pub fn get_reward(&self, arm: usize) -> usize {
        return self.rewards[arm];
    }

    pub fn get_pull(&self, arm: usize) -> usize {
        return self.pulls[arm];
    }

    pub fn pull_arm(&mut self, iteration: u32, t: u32, chosen_arm: usize) {
        assert!(iteration < self.number_of_iterations);
        assert!(chosen_arm < self.number_of_arms);

        // pull the chosen arm
        let arm: &BernoulliArm = self.arms.get(chosen_arm).unwrap();
        let reward = arm.get_reward(iteration, t);
        self.rewards[chosen_arm] += reward;
        self.pulls[chosen_arm] += 1;

        // create the new cumulative rewards for this time step
        let tot_cum = self.rewards.iter().copied().reduce(|a, b| a + b).unwrap();
        if self.total_rewards.contains_key(&t) {
            let rewards_at_time_t: &mut Vec<usize> = self.total_rewards.get_mut(&t).unwrap();
            rewards_at_time_t.push(tot_cum);
        } else {
            let mut tot_cum_vec = Vec::new();
            tot_cum_vec.push(tot_cum);
            self.total_rewards.insert(t, tot_cum_vec);
        }
    }

    pub fn get_mean_total_cumulative_rewards(&self, t: u32) -> f64 {
        let tot_cum_at_t = self.total_rewards.get(&t).unwrap();
        let res: usize = tot_cum_at_t.iter().copied().reduce(|a, b| a + b).unwrap();
        return res as f64 / tot_cum_at_t.len() as f64;
    }
}

struct SalsaExecutionTimes {
    pub setup_total: String,
    pub setup_user: String,

    pub mean_select_arm_total: String,
    pub mean_select_arm_do: String,
    pub mean_select_arm_servers: String,

    pub mean_send_rewards_total: String,
    pub mean_send_rewards_do: String,
    pub mean_send_rewards_proxy: String,
    pub mean_send_rewards_user: String,
}

fn run_multi_armed_bandits(
    arms: Vec<BernoulliArm>,
    number_of_iterations: u32,
    budget: u32,
    strategy: &mut dyn Strategy,
) -> MultiArmedBanditsExecution {
    let mut execution = MultiArmedBanditsExecution::new(arms, number_of_iterations);
    let approximation_factor = (10.0 as f64).powf(10.0) as f64;

    for iteration in 0..number_of_iterations {
        // start the execution for this iteration
        execution.start(iteration);

        // for each budget
        for t in 1..=budget {
            let mut scores = Vec::new();
            for i in 0..execution.number_of_arms() {
                let s_i = execution.get_reward(i);
                let n_i = execution.get_pull(i);
                let v_i = strategy.score(iteration, i as u32, s_i, n_i, t);
                scores.push(v_i);
            }

            // do the argmax
            let mut max_index = 0;
            let mut max_score = scores.get(max_index).unwrap().clone();
            for i in 1..scores.len() {
                let v_i = scores.get(i).unwrap().clone();
                if max_score < v_i {
                    max_index = i;
                    max_score = v_i;
                }
            }

            // pull the chosen arm
            execution.pull_arm(iteration, t, max_index);
        }
    }

    return execution;
}

use std::time::Duration;

use csv::Writer;
use salsa::run_salsa;

use mpc::*;
fn main() {
    // Extract from the MovieLens dataset
    let probs: Vec<f64> = vec![
        0.3404029692470838,
        0.05408271474019088,
        0.036055143160127257,
        0.12937433722163308,
        0.041357370095440084,
        0.015906680805938492,
        0.2788971367974549,
        0.16436903499469777,
        0.22375397667020147,
        0.06256627783669141,
        0.17709437963944857,
        0.24602332979851538,
        0.09968186638388123,
        0.14103923647932132,
        0.19936373276776245,
        0.016967126193001062,
        0.043478260869565216,
        0.003181336161187699,
        0.053022269353128315,
        0.042417815482502653,
        0.015906680805938492,
        0.24602332979851538,
        0.15482502651113467,
        0.09544008483563096,
        0.1633085896076352,
        0.031813361611876985,
        0.02332979851537646,
        0.21208907741251326,
        0.021208907741251327,
        0.031813361611876985,
        0.09862142099681867,
        0.05620360551431601,
        0.053022269353128315,
        0.0021208907741251328,
        0.0021208907741251328,
        0.0010604453870625664,
        0.0010604453870625664,
        0.031813361611876985,
        0.04029692470837752,
        0.02014846235418876,
        0.013785790031813362,
        0.10180275715800637,
        0.010604453870625663,
        0.04029692470837752,
        0.061505832449628844,
        0.016967126193001062,
        0.08377518557794274,
        0.10180275715800637,
        0.03499469777306469,
        0.5312831389183457,
        0.04772004241781548,
        0.05938494167550371,
        0.05408271474019088,
        0.042417815482502653,
        0.10286320254506894,
        0.3117709437963945,
        0.031813361611876985,
        0.10922587486744433,
        0.06998939554612937,
        0.04878048780487805,
        0.043478260869565216,
        0.04559915164369035,
        0.027571580063626724,
        0.2704135737009544,
        0.07317073170731707,
        0.09013785790031813,
        0.04559915164369035,
        0.07211028632025451,
        0.23860021208907742,
        0.16861081654294804,
        0.1420996818663839,
        0.05514316012725345,
        0.06468716861081654,
        0.003181336161187699,
        0.003181336161187699,
        0.02863202545068929,
        0.06468716861081654,
        0.006362672322375398,
        0.2799575821845175,
        0.016967126193001062,
        0.06468716861081654,
        0.17179215270413573,
        0.1474019088016967,
        0.006362672322375398,
        0.019088016967126194,
        0.11983032873807,
        0.10286320254506894,
        0.13361611876988336,
        0.23223753976670203,
        0.03711558854718982,
        0.08695652173913043,
        0.06786850477200425,
        0.0816542948038176,
        0.051961823966065745,
        0.15270413573700956,
        0.23966065747614,
        0.17709437963944857,
        0.3647932131495228,
        0.10604453870625663,
        0.4305408271474019,
    ];
    println!("inputted probs: {:?}", probs);

    let all_arms: Vec<BernoulliArm> = probs
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let mut seed = [0u8; 32];
            seed[0] = index as u8;
            BernoulliArm::new(seed, *p, index as u8)
        })
        .collect();

    println!("Running Salsa !");
    let number_of_iterations = 10;

    // UCB + N = 1000 + k = [3, 5, 7, 9] --> tango-execution-time-by-K.csv
    let mut wtr = Writer::from_path("salsa-execution-time-by-K.csv").unwrap();
    wtr.write_record(&["K", "setup_total", "select_arm_total", "send_rewards_total"])
        .unwrap();
    wtr.flush().unwrap();
    let mut strategy = UCB {};
    let budget = 1000;
    for number_of_arms in [3, 5, 7, 9] {
        let arms: Vec<BernoulliArm> = all_arms[0..number_of_arms].to_vec();
        let (execution, times) = run_salsa(arms, number_of_iterations, budget, &mut strategy);
        wtr.write_record(&[
            number_of_arms.to_string(),
            times.setup_total,
            times.mean_select_arm_total,
            times.mean_send_rewards_total,
        ])
        .unwrap();
        wtr.flush().unwrap();
    }

    // UCB + K = 9 + N = [200, 400, 600, 800, 1000] --> tango-execution-time-by-N.csv
    let mut wtr = Writer::from_path("salsa-execution-time-by-N.csv").unwrap();
    let mut wtr_totcum = Writer::from_path("salsa-ucb-totcum-by-N.csv").unwrap();
    wtr.write_record(&["N", "setup_total", "select_arm_total", "send_rewards_total"])
        .unwrap();
    wtr.flush().unwrap();
    wtr_totcum.write_record(&["N", "rewards"]).unwrap();
    wtr_totcum.flush().unwrap();

    let mut strategy = UCB {};
    let number_of_arms = 9;
    for budget in [200, 400, 600, 800, 1000] {
        let arms: Vec<BernoulliArm> = all_arms[0..number_of_arms].to_vec();
        let (execution, times) = run_salsa(arms, number_of_iterations, budget, &mut strategy);
        wtr.write_record(&[
            budget.to_string(),
            times.setup_total,
            times.mean_select_arm_total,
            times.mean_send_rewards_total,
        ])
        .unwrap();
        wtr.flush().unwrap();

        wtr_totcum
            .write_record(&[
                budget.to_string(),
                execution
                    .get_mean_total_cumulative_rewards(budget)
                    .to_string(),
            ])
            .unwrap();
        wtr_totcum.flush().unwrap();
    }
}
