#!/usr/bin/env bash

# Random planning
python -u main.py -t number_game --number_concept mul7 --sim_model memoryless --teaching_phase_actions 5 --random -- continuous > "data/ng_mul7_random-mless.log"
python -u main.py -t number_game --number_concept mul7 --sim_model discrete --teaching_phase_actions 5 --random -- continuous > "data/ng_mul7_random-discrete.log"
python -u main.py -t number_game --number_concept mul7 --sim_model continuous --teaching_phase_actions 5 --random -- continuous > "data/ng_mul7_random-continuous.log"


# Random QE planning
python -u main.py -t number_game --number_concept mul7 --sim_model memoryless --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_mul7_random_qe-mless.log"
python -u main.py -t number_game --number_concept mul7 --sim_model discrete --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_mul7_random_qe-discrete.log"
python -u main.py -t number_game --number_concept mul7 --sim_model continuous --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_mul7_random_qe-continuous.log"


# MIG planning
python -u main.py -t number_game --number_concept mul7 --sim_model memoryless --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_mul7_mig-mless.log"
python -u main.py -t number_game --number_concept mul7 --sim_model discrete --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_mul7_mig-discrete.log"
python -u main.py -t number_game --number_concept mul7 --sim_model continuous --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_mul7_mig-continuous.log"


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



# Random planning
python -u main.py -t number_game --number_concept mul4-1 --sim_model memoryless --teaching_phase_actions 5 --random -- continuous > "data/ng_mul4-1_random-mless.log"
python -u main.py -t number_game --number_concept mul4-1 --sim_model discrete --teaching_phase_actions 5 --random -- continuous > "data/ng_mul4-1_random-discrete.log"
python -u main.py -t number_game --number_concept mul4-1 --sim_model continuous --teaching_phase_actions 5 --random -- continuous > "data/ng_mul4-1_random-continuous.log"


# Random QE planning
python -u main.py -t number_game --number_concept mul4-1 --sim_model memoryless --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_mul4-1_random_qe-mless.log"
python -u main.py -t number_game --number_concept mul4-1 --sim_model discrete --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_mul4-1_random_qe-discrete.log"
python -u main.py -t number_game --number_concept mul4-1 --sim_model continuous --teaching_phase_actions 5 --random --actions_qe_only -- continuous > "data/ng_mul4-1_random_qe-continuous.log"


# MIG planning
python -u main.py -t number_game --number_concept mul4-1 --sim_model memoryless --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_mul4-1_mig-mless.log"
python -u main.py -t number_game --number_concept mul4-1 --sim_model discrete --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_mul4-1_mig-discrete.log"
python -u main.py -t number_game --number_concept mul4-1 --sim_model continuous --teaching_phase_actions 5 --plan_max_gain -- continuous > "data/ng_mul4-1_mig-continuous.log"

