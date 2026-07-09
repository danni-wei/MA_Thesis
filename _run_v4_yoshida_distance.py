import sys; sys.path.insert(0, '.')
from pinn_hmc.experiment_yoshida_distance_adaptive import main

full = "--full" in sys.argv
main(n_rep=30 if full else 2)
