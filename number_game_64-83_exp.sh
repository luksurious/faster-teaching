#!/usr/bin/env bash

# Memoryless planning
python -u main.py -t number_game --number_concept 64-83 --sim_model memoryless --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 8 -- memoryless > "data/ng_64-83_mless-mless.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model discrete --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 8 --plan_load_actions data/actions.pickle -- memoryless > "data/ng_64-83_mless-discrete.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model continuous --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 8 --plan_load_actions data/actions.pickle -- memoryless > "data/ng_64-83_mless-continuous.log"


# Discrete planning
python -u main.py -t number_game --number_concept 64-83 --sim_model memoryless --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 6 -- discrete > "data/ng_64-83_discrete-mless.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model discrete --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 6 --plan_load_actions data/actions.pickle -- discrete > "data/ng_64-83_discrete-discrete.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model continuous --plan_pre_steps 20 --plan_pre_horizon 2 --teaching_phase_actions 5 --plan_online_horizon 2 --plan_online_samples 6 6 --plan_load_actions data/actions.pickle -- discrete > "data/ng_64-83_discrete-continuous.log"


# Continuous planning
python -u main.py -t number_game --number_concept 64-83 --sim_model memoryless --plan_pre_steps 20 --plan_pre_horizon 3 --teaching_phase_actions 5 --plan_online_horizon 3 --plan_online_samples 6 6 8 -- continuous > "data/ng_64-83_continuous-mless.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model discrete --plan_pre_steps 20 --plan_pre_horizon 3 --teaching_phase_actions 5 --plan_online_horizon 3 --plan_online_samples 6 6 8 --plan_load_actions data/actions.pickle -- continuous > "data/ng_64-83_continuous-discrete.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model continuous --plan_pre_steps 20 --plan_pre_horizon 3 --teaching_phase_actions 5 --plan_online_horizon 3 --plan_online_samples 6 6 8 --plan_load_actions data/actions.pickle -- continuous > "data/ng_64-83_continuous-continuous.log"



# Random planning
python -u main.py -t number_game --number_concept 64-83 --sim_model memoryless --teaching_phase_actions 5 --random -- continuous > "data/ng_64-83_random-mless.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model discrete --teaching_phase_actions 5 --random -- continuous > "data/ng_64-83_random-discrete.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model continuous --teaching_phase_actions 5 --random -- continuous > "data/ng_64-83_random-continuous.log"


# Random QE planning
python -u main.py -t number_game --number_concept 64-83 --sim_model memoryless --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_64-83_random_qe-mless.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model discrete --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_64-83_random_qe-discrete.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model continuous --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_64-83_random_qe-continuous.log"


# MIG planning
python -u main.py -t number_game --number_concept 64-83 --sim_model memoryless --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_64-83_mig-mless.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model discrete --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_64-83_mig-discrete.log"
python -u main.py -t number_game --number_concept 64-83 --sim_model continuous --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_64-83_mig-continuous.log"