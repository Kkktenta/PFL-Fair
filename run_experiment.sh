./scripts/run_fedavg_adult.sh
./scripts/run_fairfed_adult.sh
./scripts/run_fedala_adult.sh
./scripts/run_pflfair_adult.sh

conda run -n PFL-Fair python3 ./scripts/visualize_results.py