#!/usr/bin/env bash

COMMON_ARGS='--plan_pre_steps 9 --plan_pre_horizon 2 --plan_online_horizon 2 '

# Memoryless planning
python -u main.py $COMMON_ARGS --sim_model memoryless --plan_online_samples 7 6 -- memoryless letter > "data/letter_mless-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --plan_online_samples 7 6 --plan_load_actions data/actions.pickle -- memoryless letter > "data/letter_mless-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --plan_online_samples 7 6 --plan_load_actions data/actions.pickle -- memoryless letter > "data/letter_mless-continuous.log"


# Discrete planning
python -u main.py $COMMON_ARGS --sim_model memoryless --plan_online_samples 8 8 -- discrete letter > "data/letter_discrete-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --plan_online_samples 8 8 --plan_load_actions data/actions.pickle -- discrete letter > "data/letter_discrete-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --plan_online_samples 8 8 --plan_load_actions data/actions.pickle -- discrete letter > "data/letter_discrete-continuous.log"


# Continuous planning
python -u main.py $COMMON_ARGS --sim_model memoryless --plan_online_samples 4 3 -- continuous letter > "data/letter_continuous-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --plan_online_samples 4 3 --plan_load_actions data/actions.pickle -- continuous letter > "data/letter_continuous-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --plan_online_samples 4 3 --plan_load_actions data/actions.pickle -- continuous letter > "data/letter_continuous-continuous.log"


# Random planning
python -u main.py $COMMON_ARGS --sim_model memoryless -- random letter > "data/letter_random-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete -- random letter > "data/letter_random-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous -- random letter > "data/letter_random-continuous.log"


# Random QE planning
python -u main.py $COMMON_ARGS --sim_model memoryless --actions_qe_only -- random letter > "data/letter_random_qe-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete --actions_qe_only -- random letter > "data/letter_random_qe-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous --actions_qe_only -- random letter > "data/letter_random_qe-continuous.log"


# MIG planning
python -u main.py $COMMON_ARGS --sim_model memoryless -- mig letter > "data/letter_mig-mless.log"
python -u main.py $COMMON_ARGS --sim_model discrete -- mig letter > "data/letter_mig-discrete.log"
python -u main.py $COMMON_ARGS --sim_model continuous -- mig letter > "data/letter_mig-continuous.log"

