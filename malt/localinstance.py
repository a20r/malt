
import localization
import matplotlib.pyplot as plt
import numpy as np


class LocalInstance(object):

    def __init__(self, r_ref, l_ref, pos_sim, node_events, sim):
        self.r_ref = r_ref
        self.l_ref = l_ref
        self.node_events = node_events
        self.x_dim = sim.x_dim
        self.y_dim = sim.y_dim
        self.sim = sim
        self.v_num = 100
        self.pos_sim = pos_sim
        self.locations = None
        self.max_location = None

    def get_detection_events(self):
        return self.node_events

    def localize(self):
        return localization.determine_source_locations_instance(
            self.r_ref, self.l_ref, self.node_events, disp=0
        )

    def get_locations(self):
        if self.locations is None:
            self.locations = self.localize()

        return self.locations

    def get_max_location(self):
        locations = self.get_locations()
        max_local = None
        if self.max_location is None:
            for local in locations:
                if max_local is None or\
                        local.get_confidence() > max_local.get_confidence():
                    max_local = local
            self.max_local = max_local

        return self.max_local

    def get_error(self):
        max_local = self.get_max_location()
        return max_local.get_position().dist_to(self.pos_sim)

    def plot(self):
        locations = self.get_locations()
        d_events = self.node_events
        fig = plt.figure("")
        ax = fig.add_subplot(111)
        ax.set_title("Source location probability density function")
        ax.set_xlabel("X Location")
        ax.set_ylabel("Y Location")

        x_step = self.x_dim / self.v_num
        y_step = self.y_dim / self.v_num
        x = np.arange(0, self.x_dim, x_step)
        y = np.arange(0, self.y_dim, y_step)
        X, Y = np.meshgrid(x, y)
        zs = np.array(
            [
                localization.position_probability(
                    x_i, y_i, self.r_ref, self.l_ref, d_events
                )
                for x_i, y_i in zip(np.ravel(X), np.ravel(Y))
            ]
        )

        Z = zs.reshape(X.shape)

        ax.pcolormesh(X, Y, Z, shading="interp")
        ax.scatter(
            [p.position.x for p in locations],
            [p.position.y for p in locations],
            marker="+",
            linewidths=15,
            c="white"
        )
        ax.scatter(
            [d_event.x for d_event in d_events],
            [d_event.y for d_event in d_events],
            marker="o",
            linewidths=5,
            c="white",
            s=300
        )
        ax.scatter(
            [self.pos_sim.x],
            [self.pos_sim.y],
            marker="x",
            linewidths=5,
            c="black",
            s=300
        )

        ax.set_xlim(0, self.x_dim)
        ax.set_ylim(0, self.y_dim)
        plt.show()

        return locations
