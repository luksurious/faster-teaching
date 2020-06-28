### Replication of
# Faster Teaching via POMDP Planning
_by Rafferty et. al (2016)_ (https://www.onlinelibrary.wiley.com/doi/full/10.1111/cogs.12290)


**Authors**: Lukas Brückner, Aurélien Nioche (User Interfaces Group, Aalto University, Finland)

[Replication paper](replication-paper.pdf)

## Requirements
* Python 3 (tested with 3.7+)
* Modules as listed in `requirements.txt`

## Setup
1. (Optional) Create virtual/conda environment
2. Install required modules via `pip install -r requirements.txt`
3. Create `data` folder

## Running

`python main.py [options] <policy> <task>`

*Full list of options*

```
usage: main.py [-h|--help]
               # general config
               [-v|--verbose]
               [--no_show]
               
               # learner
               [-m|--manual]
               [-s|--single]
               [-c SIM_COUNT]
               [--pool POOL]
               # learner simulation options
               [--sim_seed SIM_SEED]
               [--sim_model {memoryless,discrete,continuous}]
               [--sim_discrete_memory SIM_DISCRETE_MEMORY]
               [--sim_pause SIM_PAUSE]
               [--sim_no_noise]

               # number game options
               [--number_concept {mul7,64-83,mul4-1}]

               # letter arithmetic options
               [-l|--problem_len PROBLEM_LEN]
               [-r|--number_range NUMBER_RANGE]

               # teacher options
               [--teaching_phase_actions TEACHING_PHASE_ACTIONS]
               [--max_teaching_phases MAX_TEACHING_PHASES]

               # policy options
               [--actions_qe_only]
               [--plan_no_noise]
               [--plan_discrete_memory PLAN_DISCRETE_MEMORY]      
               [--particle_limit PARTICLE_LIMIT] 
    
               [--plan_online_horizon PLAN_ONLINE_HORIZON]
               [--plan_online_samples [PLAN_ONLINE_SAMPLES [...]]]

               [--plan_pre_steps PLAN_PRE_STEPS]
               [--plan_pre_horizon PLAN_PRE_HORIZON]
               [--plan_pre_samples PLAN_PRE_SAMPLES]
               [--plan_load_actions PLAN_LOAD_ACTIONS]

               # policy
               [{memoryless,discrete,continuous,random,mig}]
               # task
               {letter,number_game}

```

Experiments from the paper are listed in the varios `*.sh` shell scripts.



**Examples**

* Manual learner via CLI with the letter arithmetic task and a random policy (verbose)
  `python main.py -m -v random letter`
* Single simulated memoryless learner of the number game (task multiples of 7) with the memoryless policy (verbose)
  `python main.py -sv --sim_model memoryless --number_concept mul7 memoryless number_game`
* 50 simulated trials of the continuous learner with the discrete memory policy for the letter arithmetic task with precomputation and defined horizon sampling
  `python main.py -c --plan_pre_steps 9 --plan_pre_horizon 2 --plan_online_horizon 2  --sim_model continuous --plan_online_samples 8 8 -- discrete letter`

JSON output and generated figures are stored in the `data` folder.



