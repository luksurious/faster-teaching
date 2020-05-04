#!/usr/bin/env bash

COMMON_ARGS='--number_concept mul7 --plan_pre_steps 20 --teaching_phase_actions 5 '

# Memoryless planning
python -u main.py $COMMON_ARGS --sim_model memoryless --plan_pre_horizon 2 --plan_online_horizon 2 --plan_online_samples 6 8 -- memoryless number_game > "data/ng_mul7_mless-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --plan_pre_horizon 2 --plan_online_horizon 2 --plan_online_samples 6 8 --plan_load_actions data/actions.pickle -- memoryless number_game > "data/ng_mul7_mless-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --plan_pre_horizon 2 --plan_online_horizon 2 --plan_online_samples 6 8 --plan_load_actions data/actions.pickle -- memoryless number_game > "data/ng_mul7_mless-continuous.log"


# Discrete planning
python -u main.py $COMMON_ARGS --sim_model memoryless --plan_pre_horizon 2 --plan_online_horizon 2 --plan_online_samples 6 6 -- discrete number_game > "data/ng_mul7_discrete-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --plan_pre_horizon 2 --plan_online_horizon 2 --plan_online_samples 6 6 --plan_load_actions data/actions.pickle -- discrete number_game > "data/ng_mul7_discrete-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --plan_pre_horizon 2 --plan_online_horizon 2 --plan_online_samples 6 6 --plan_load_actions data/actions.pickle -- discrete number_game > "data/ng_mul7_discrete-continuous.log"


# Continuous planning
python -u main.py $COMMON_ARGS --sim_model memoryless --plan_pre_horizon 3 --plan_online_horizon 3 --plan_online_samples 6 6 8 -- continuous number_game > "data/ng_mul7_continuous-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --plan_pre_horizon 3 --plan_online_horizon 3 --plan_online_samples 6 6 8 --plan_load_actions data/actions.pickle -- continuous number_game > "data/ng_mul7_continuous-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --plan_pre_horizon 3 --plan_online_horizon 3 --plan_online_samples 6 6 8 --plan_load_actions data/actions.pickle -- continuous number_game > "data/ng_mul7_continuous-continuous.log"


# Random planning
python -u main.py $COMMON_ARGS --sim_model memoryless -- random number_game > "data/ng_mul7_random-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete -- random number_game > "data/ng_mul7_random-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous -- random number_game > "data/ng_mul7_random-continuous.log"


# Random QE planning
python -u main.py $COMMON_ARGS --sim_model memoryless --actions_qe_only -- random number_game > "data/ng_mul7_random_qe-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --actions_qe_only -- random number_game > "data/ng_mul7_random_qe-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --actions_qe_only -- random number_game > "data/ng_mul7_random_qe-continuous.log"


# MIG planning
python -u main.py $COMMON_ARGS --sim_model memoryless -- mig number_game > "data/ng_mul7_mig-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete -- mig number_game > "data/ng_mul7_mig-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous -- mig number_game > "data/ng_mul7_mig-continuous.log"

