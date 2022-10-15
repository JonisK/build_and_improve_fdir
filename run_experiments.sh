#!/bin/bash

rm -rf experiment_logs
mkdir experiment_logs
mkdir experiment_logs/space_tug_v6
mkdir experiment_logs/space_tug_v7
mkdir experiment_logs/earth_observation_v4


python3 src/mcts.py benchmarks/space_tug_v6/space_tug_v6.dot --modecosts benchmarks/space_tug_v6/space_tug_v6_mode_costs.txt --equipfailprobs benchmarks/space_tug_v6/space_tug_v6_fault_probabilities.txt --successorstokeep 1 --simulationsize 200 --mctsstrat --evaluatenaive > experiment_logs/space_tug_v6/mcts.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/space_tug_v6/space_tug_v6.dot --modecosts benchmarks/space_tug_v6/space_tug_v6_mode_costs.txt --equipfailprobs benchmarks/space_tug_v6/space_tug_v6_fault_probabilities.txt --successorstokeep 2 --simulationsize 200 > experiment_logs/space_tug_v6/prism_2.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/space_tug_v6/space_tug_v6.dot --modecosts benchmarks/space_tug_v6/space_tug_v6_mode_costs.txt --equipfailprobs benchmarks/space_tug_v6/space_tug_v6_fault_probabilities.txt --successorstokeep 10 --simulationsize 200 > experiment_logs/space_tug_v6/prism_10.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/space_tug_v6/space_tug_v6.dot --modecosts benchmarks/space_tug_v6/space_tug_v6_mode_costs.txt --equipfailprobs benchmarks/space_tug_v6/space_tug_v6_fault_probabilities.txt --successorstokeep 0 --simulationsize 0 > experiment_logs/space_tug_v6/prism_all.txt
echo $'\n\n\n'
echo $'\n\n\n'




python3 src/mcts.py benchmarks/space_tug_v7/space_tug_v7.dot --modecosts benchmarks/space_tug_v7/space_tug_v7_mode_costs.txt --equipfailprobs benchmarks/space_tug_v7/space_tug_v7_fault_probabilities.txt --successorstokeep 1 --simulationsize 200 --mctsstrat --evaluatenaive > experiment_logs/space_tug_v7/mcts.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/space_tug_v7/space_tug_v7.dot --modecosts benchmarks/space_tug_v7/space_tug_v7_mode_costs.txt --equipfailprobs benchmarks/space_tug_v7/space_tug_v7_fault_probabilities.txt --successorstokeep 2 --simulationsize 200 > experiment_logs/space_tug_v7/prism_2.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/space_tug_v7/space_tug_v7.dot --modecosts benchmarks/space_tug_v7/space_tug_v7_mode_costs.txt --equipfailprobs benchmarks/space_tug_v7/space_tug_v7_fault_probabilities.txt --successorstokeep 10 --simulationsize 200 > experiment_logs/space_tug_v7/prism_10.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/space_tug_v7/space_tug_v7.dot --modecosts benchmarks/space_tug_v7/space_tug_v7_mode_costs.txt --equipfailprobs benchmarks/space_tug_v7/space_tug_v7_fault_probabilities.txt --successorstokeep 0 --simulationsize 0 > experiment_logs/space_tug_v7/prism_all.txt
echo $'\n\n\n'
echo $'\n\n\n'




python3 src/mcts.py benchmarks/earth_observation_v4/earth_observation_v4.dot --modecosts benchmarks/earth_observation_v4/earth_observation_v4_mode_costs.txt --equipfailprobs benchmarks/earth_observation_v4/earth_observation_v4_fault_probabilities.txt --successorstokeep 1 --simulationsize 200 --mctsstrat --evaluatenaive > experiment_logs/earth_observation_v4/mcts.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/earth_observation_v4/earth_observation_v4.dot --modecosts benchmarks/earth_observation_v4/earth_observation_v4_mode_costs.txt --equipfailprobs benchmarks/earth_observation_v4/earth_observation_v4_fault_probabilities.txt --successorstokeep 2 --simulationsize 200 > experiment_logs/earth_observation_v4/prism_2.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/earth_observation_v4/earth_observation_v4.dot --modecosts benchmarks/earth_observation_v4/earth_observation_v4_mode_costs.txt --equipfailprobs benchmarks/earth_observation_v4/earth_observation_v4_fault_probabilities.txt --successorstokeep 10 --simulationsize 200 > experiment_logs/earth_observation_v4/prism_10.txt
echo $'\n\n\n'
python3 src/mcts.py benchmarks/earth_observation_v4/earth_observation_v4.dot --modecosts benchmarks/earth_observation_v4/earth_observation_v4_mode_costs.txt --equipfailprobs benchmarks/earth_observation_v4/earth_observation_v4_fault_probabilities.txt --successorstokeep 0 --simulationsize 0 > experiment_logs/earth_observation_v4/prism_all.txt

