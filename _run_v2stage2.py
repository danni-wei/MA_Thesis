import sys; sys.path.insert(0, '.')
from pinn_hmc.experiment_sus_stage2 import main

full = "--full" in sys.argv
main(n_rep=30 if full else 2)
