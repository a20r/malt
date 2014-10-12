
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import malt
import random
from clint.textui import progress
import time

r_ref = 1
l_ref = 100
num_nodes = [10, 20, 30, 40, 50, 60]
num_runs = 50
num_iters = 400
out_file = "data/{}_{}_{}.txt"


def experiment_nn(dim):
    for nn in progress.bar(num_nodes):
        for i in xrange(num_runs):
            sim = malt.Simulation(
                x_dim=dim,
                y_dim=dim,
                randomly_generate=True,
                num_nodes=nn,
                scene="scenes/{}.out".format(dim)
            )

            x_stim = random.randint(0, dim)
            y_stim = random.randint(0, dim)

            filename = out_file.format(dim, nn, i)

            with open(filename, "w") as f:
                for j in xrange(num_iters):
                    local_instance = sim.step(x_stim, y_stim, r_ref, l_ref)
                    f.write("{} {}\n".format(
                        local_instance.get_error(),
                        time.time()
                    ))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_nn(int(sys.argv[1]))
    else:
        print "Please enter a dimension"
