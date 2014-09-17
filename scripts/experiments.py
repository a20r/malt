
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import malt
import random
import numpy as np


r_ref = 1
l_ref = 50
num_nodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
intensity_noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
num_runs = 50


def experiment_nn():
    out_file = "sandbox/nn.txt"
    with open(out_file, "w") as f:
        for nn in num_nodes:
            errs = list()
            avg_err = 0.0
            for i in xrange(num_runs):
                sim = malt.Simulation(
                    x_dim=100,
                    y_dim=100,
                    randomly_generate=True,
                    num_nodes=nn,
                    time_noise=2,
                    intensity_noise=0.1,
                    confidence_noise=0.1
                )

                x_stim = random.randint(0, 100)
                y_stim = random.randint(0, 100)

                local_instance = sim.provide_stimulus(
                    x_stim, y_stim, r_ref, l_ref
                )

                err = local_instance.get_error()
                avg_err += err / float(num_runs)
                errs.append(err)

                print "NN:", nn,
                print "|| Iter:", i,
                print "|| Error:", err, "meters"

            std_err = np.std(errs)
            f.write("{} {} {}\n".format(nn, avg_err, std_err))


def experiment_intensity_noise():
    out_file = "sandbox/in_{}.txt"
    for nn in [10, 20, 30]:
        with open(out_file.format(nn), "w") as f:
            for intensity_noise in intensity_noises:
                errs = list()
                avg_err = 0.0
                for i in xrange(num_runs):
                    sim = malt.Simulation(
                        x_dim=100,
                        y_dim=100,
                        randomly_generate=True,
                        num_nodes=nn,
                        time_noise=2,
                        intensity_noise=intensity_noise,
                        confidence_noise=0.1
                    )

                    x_stim = random.randint(0, 100)
                    y_stim = random.randint(0, 100)

                    local_instance = sim.provide_stimulus(
                        x_stim, y_stim, r_ref, l_ref
                    )

                    err = local_instance.get_error()
                    avg_err += err / float(num_runs)
                    errs.append(err)

                    print "NN:", nn,
                    print "|| Iter:", i,
                    print " || Error:", err, "meters"

                std_err = np.std(errs)
                f.write("{} {} {}\n".format(intensity_noise, avg_err, std_err))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--nn":
            experiment_nn()
        elif sys.argv[1] == "--in":
            experiment_intensity_noise()
        else:
            raise RuntimeError("Arguments provided suck ass")
    else:
        raise RuntimeError("You didnt provide arguments dipshit")
