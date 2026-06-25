import sys; sys.path.insert(0, '.')
from pinn_hmc.experiment_geometry_bounce import main

full      = "--full"      in sys.argv
skip_exp0 = "--skip-exp0" in sys.argv
main(n_rep=30 if full else 2, skip_exp0=skip_exp0)
