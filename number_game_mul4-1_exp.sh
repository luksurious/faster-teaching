#!/usr/bin/env bash

# Ensure target folder of log files exists
mkdir -p data

# Memoryless planning
python -u main.py --number_concept mul4-1 --sim_model memoryless --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 8 -- memoryless number_game > "data/ng_mul4-1_mless-mless.log"
python -u main.py --number_concept mul4-1 --sim_model discrete --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 8 --plan_load_actions data/actions.pickle -- memoryless number_game > "data/ng_mul4-1_mless-discrete.log"
python -u main.py --number_concept mul4-1 --sim_model continuous --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 8 --plan_load_actions data/actions.pickle -- memoryless number_game > "data/ng_mul4-1_mless-continuous.log"


# Discrete planning
python -u main.py --number_concept mul4-1 --sim_model memoryless --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 6 -- discrete number_game > "data/ng_mul4-1_discrete-mless.log"
python -u main.py --number_concept mul4-1 --sim_model discrete --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 6 --plan_load_actions data/actions.pickle -- discrete number_game > "data/ng_mul4-1_discrete-discrete.log"
python -u main.py --number_concept mul4-1 --sim_model continuous --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 6 --plan_load_actions data/actions.pickle -- discrete number_game > "data/ng_mul4-1_discrete-continuous.log"


# Continuous planning
python -u main.py --number_concept mul4-1 --sim_model memoryless --plan_pre_steps 20 --plan_pre_horizon 3 --teaching_phase_actions 5 --plan_online_horizon 3 --plan_online_samples 6 6 8 -- continuous number_game > "data/ng_mul4-1_continuous-mless.log"
python -u main.py --number_concept mul4-1 --sim_model discrete --plan_pre_steps 20 --plan_pre_horizon 3 --teaching_phase_actions 5 --plan_online_horizon 3 --plan_online_samples 6 6 8 --plan_load_actions data/actions.pickle -- continuous number_game > "data/ng_mul4-1_continuous-discrete.log"
python -u main.py --number_concept mul4-1 --sim_model continuous --plan_pre_steps 20 --plan_pre_horizon 3 --teaching_phase_actions 5 --plan_online_horizon 3 --plan_online_samples 6 6 8 --plan_load_actions data/actions.pickle -- continuous number_game > "data/ng_mul4-1_continuous-continuous.log"



# Random planning
python -u main.py --number_concept mul4-1 --sim_model memoryless --teaching_phase_actions 5 -- random number_game > "data/ng_mul4-1_random-mless.log"
python -u main.py --number_concept mul4-1 --sim_model discrete --teaching_phase_actions 5 -- random number_game > "data/ng_mul4-1_random-discrete.log"
python -u main.py --number_concept mul4-1 --sim_model continuous --teaching_phase_actions 5 -- random number_game > "data/ng_mul4-1_random-continuous.log"


# Random QE planning
python -u main.py --number_concept mul4-1 --sim_model memoryless --teaching_phase_actions 5 --actions_qe_only -- random number_game > "data/ng_mul4-1_random_qe-mless.log"
python -u main.py --number_concept mul4-1 --sim_model discrete --teaching_phase_actions 5 --actions_qe_only -- random number_game > "data/ng_mul4-1_random_qe-discrete.log"
python -u main.py --number_concept mul4-1 --sim_model continuous --teaching_phase_actions 5 --actions_qe_only -- random number_game > "data/ng_mul4-1_random_qe-continuous.log"


# MIG planning
python -u main.py --number_concept mul4-1 --sim_model memoryless --teaching_phase_actions 5 -- mig number_game > "data/ng_mul4-1_mig-mless.log"
python -u main.py --number_concept mul4-1 --sim_model discrete --teaching_phase_actions 5 -- mig number_game > "data/ng_mul4-1_mig-discrete.log"
python -u main.py --number_concept mul4-1 --sim_model continuous --teaching_phase_actions 5 -- mig number_game > "data/ng_mul4-1_mig-continuous.log"
