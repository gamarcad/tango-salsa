use crate::{BernoulliArm, MultiArmedBanditsExecution, SalsaExecutionTimes, Strategy};
use std::time::{Duration, Instant};

use crate::mpc;
use crate::mpc::{add_arithmetic, reconstruct_arithmetic, share_arithmetic, share_u64};

use std::hint::black_box;

pub(crate) fn run_salsa(
    arms: Vec<BernoulliArm>,
    number_of_iterations: u32,
    budget: u32,
    strategy: &mut dyn Strategy,
) -> (MultiArmedBanditsExecution, SalsaExecutionTimes) {
    // create the execution time registers
    let mut execution = MultiArmedBanditsExecution::new(arms, number_of_iterations);
    let k: u64 = execution.number_of_arms() as u64;
    let approximation_factor = (10.0 as f64).powf(10.0) as f64;

    // we set the timing variables
    let mut select_arm_total: Duration = Duration::ZERO;
    let mut send_rewards_total: Duration = Duration::ZERO;

    // we setup the public keys
    // ... for the user ...
    let start = Instant::now();
    //let (pk_ahe, sk_ahe)  = Paillier::keypair().keys();
    let setup_total = start.elapsed();

    println!("Keys generated, running...");
    for iteration in 0..number_of_iterations {
        // start the execution for this iteration
        execution.start(iteration);

        // for each budget
        for t in 1..=budget {
            let select_arm_start = Instant::now();

            // Data Owner -----------------------------------------------------------------------------------------
            //let start = Instant::now();

            // each data owner computes its own score
            let mut approximated_scores = Vec::new();
            let mut real_scores = Vec::new();
            for i in 0..execution.number_of_arms() {
                let s_i = execution.get_reward(i);
                let n_i = execution.get_pull(i);
                let score = strategy.score(iteration, i as u32, s_i, n_i, t);

                // save the real score, just to confirm the compatibility of our solution
                real_scores.push(score);

                let v_i = (approximation_factor * score).round() as u64;
                approximated_scores.push(v_i);
            }

            // each data owner shares its own scores
            let shared_approximated_scores: Vec<Vec<(u8, u8)>> = approximated_scores
                .iter()
                .map(|i| share_u64(i.clone()))
                .collect();

            // each data owner encrypts its cumulative rewards
            /*
            let indexes : Vec<usize> = (0..execution.number_of_arms()).collect();
            let encrypted_rewards : Vec<EncodedCiphertext<u64>> = indexes.iter()
                .map(|&index| {
                    paillier::Paillier::encrypt( &pk_ahe, execution.get_reward(index) as u64 )
                }).collect();
            */

            let indexes: Vec<usize> = (0..execution.number_of_arms()).collect();
            let shared_rewards: Vec<(i64, i64)> = indexes
                .iter()
                .map(|&index| share_arithmetic(execution.get_reward(index) as u64))
                .collect();
            black_box(shared_rewards);

            //select_arm_do += start.elapsed().as_millis();

            /*
            // sanity check: ensures that the approximated score does not affect the
            let mut best_real_score_index = 0;
            let mut best_approximate_score_index = 0;
            let mut best_real_score = *real_scores.first().unwrap();
            let mut best_approximate_score = *approximated_scores.first().unwrap();
            for i in 0..execution.number_of_arms() {
                let real_score = real_scores.get(i).unwrap();
                let approximate_score = approximated_scores.get(i).unwrap();
                if  best_real_score < *real_score  {
                    best_real_score = *real_score;
                    best_real_score_index = i;
                }
                if best_approximate_score < *approximate_score {
                    best_approximate_score = *approximate_score;
                    best_approximate_score_index = i;
                }
            }*/

            // the servers does the two-party computation of the argmax
            //let start = Instant::now();

            let shared_index = mpc::argmax(shared_approximated_scores);
            let shared_selection_bits = mpc::selection_bits(shared_index, k as usize);

            assert_eq!(shared_selection_bits.len(), execution.number_of_arms);

            //select_arm_servers += start.elapsed().as_millis();

            // Data Owner ------------------------------------------------------------------------------------------------------------------
            //let start = Instant::now();

            // recover the selection bits
            for (index, &shared_selection_bit) in shared_selection_bits.iter().enumerate() {
                let selection_bit = mpc::reconstruct(shared_selection_bit);

                if selection_bit == 1u8 {
                    // pull the chosen arm
                    execution.pull_arm(iteration, t, index);
                }
            }

            //select_arm_do += start.elapsed().as_millis();

            // end - start
            let end = select_arm_start.elapsed();
            select_arm_total += end;
            //#[cfg(feature = "debug")]
            println!(
                "[Salsa, iteration = {}/{}, N = {}, K = {}, t = {}]: {}ms or {}microseconds",
                iteration,
                number_of_iterations,
                budget,
                execution.number_of_arms(),
                t,
                end.as_millis(),
                end.as_micros()
            );
        }

        // Data Owner ------------------------------------------
        let start = Instant::now();

        let indexes: Vec<usize> = (0..execution.number_of_arms()).collect();
        let shared_rewards: Vec<(i64, i64)> = indexes
            .iter()
            .map(|&index| share_arithmetic(execution.get_reward(index) as u64))
            .collect();

        //send_rewards_do += start.elapsed().as_millis();

        // Proxy  ------------------------------------------
        //let start = Instant::now();

        let mut share_totcum: (i64, i64) = shared_rewards.first().unwrap().clone();
        for i in 1..shared_rewards.len() {
            let shared_reward = shared_rewards.get(i).unwrap().clone();
            share_totcum = add_arithmetic(share_totcum, shared_reward);
        }

        //send_rewards_proxy += start.elapsed().as_millis();

        // User --------------------------------------------------
        let start = Instant::now();

        let res = reconstruct_arithmetic(share_totcum);
        let expected_res = (0..k)
            .map(|i| execution.get_reward(i as usize))
            .reduce(|a, b| a + b)
            .unwrap();
        assert_eq!(res, expected_res as u64);

        send_rewards_total += start.elapsed();
    }

    // we set the timing variables
    let number_of_arms = execution.number_of_arms();

    let mean_setup_total: f64 = (setup_total.as_millis() as f64);
    let mean_select_arm_total: f64 =
        (select_arm_total.as_millis() as f64 / number_of_iterations as f64);
    let mean_send_rewards_total: f64 =
        (send_rewards_total.as_millis() as f64 / number_of_iterations as f64);

    return (
        execution,
        SalsaExecutionTimes {
            setup_total: mean_setup_total.to_string(),
            setup_user: 0.to_string(), //setup_user.to_string(),

            mean_select_arm_total: mean_select_arm_total.to_string(), //(mean_select_arm_do +  mean_select_arm_servers).to_string(),
            mean_select_arm_do: 0.to_string(), //mean_select_arm_do.to_string(),
            mean_select_arm_servers: 0.to_string(), //mean_select_arm_servers.to_string(),

            mean_send_rewards_total: mean_send_rewards_total.to_string(), //: (mean_send_rewards_do + mean_send_rewards_proxy + mean_send_rewards_user).to_string(),
            mean_send_rewards_do: 0.to_string(), //mean_send_rewards_do.to_string(),
            mean_send_rewards_proxy: 0.to_string(), // mean_send_rewards_proxy.to_string(),
            mean_send_rewards_user: 0.to_string(), // mean_send_rewards_user.to_string()
        },
    );
}
