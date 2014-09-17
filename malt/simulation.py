
import point
import random
import localization
import time
import matplotlib.pyplot as plt
import numpy as np
from detectionevent import DetectionEvent


class Simulation(object):

    def __init__(self, **kwargs):
        self.node_positions = kwargs.get("node_positions", list())
        self.intensity_noise = kwargs.get("intensity_noise", 0)
        self.position_noise = kwargs.get("position_noise", 0)
        self.confidence_noise = kwargs.get("confidence_noise", 0)
        self.time_noise = kwargs.get("time_noise", 0)
        self.randomly_generate = kwargs.get("randomly_generate", False)
        self.num_nodes = kwargs.get("num_nodes", 0)
        self.x_dim = kwargs.get("x_dim", 0)
        self.y_dim = kwargs.get("y_dim", 0)
        self.kwargs = kwargs
        self.init_nodes()

    def init_nodes(self):
        if self.randomly_generate:
            for _ in xrange(self.num_nodes):
                n_pos = point.get_random_point(self.x_dim, self.y_dim)
                self.node_positions.append(n_pos)

    def provide_stimulus(self, x, y, r_ref, l_ref):
        pos = point.Point(x, y)
        node_events = list()
        for node_pos in self.node_positions:
            dist = pos.dist_to(node_pos)

            intensity_noise = random.gauss(0, self.intensity_noise)
            intensity = localization.intensity_at_distance(
                r_ref, l_ref, dist
            ) + intensity_noise
            node_intensity = intensity_noise * intensity + intensity

            # print 100 * abs(node_intensity - intensity) / intensity

            confidence = 1 - random.gauss(0, self.confidence_noise)
            timestamp = time.time() - random.gauss(0, self.time_noise)
            x_pos = node_pos.get_x() + random.gauss(0, self.position_noise)
            y_pos = node_pos.get_y() + random.gauss(0, self.position_noise)

            node_events.append(DetectionEvent(
                x_pos, y_pos,
                confidence,
                node_intensity,
                timestamp
            ))

        return LocalInstance(r_ref, l_ref, pos, node_events, **self.kwargs)


class LocalInstance(object):

    def __init__(self, r_ref, l_ref, pos_sim, node_events, **kwargs):
        self.r_ref = r_ref
        self.l_ref = l_ref
        self.node_events = node_events
        self.x_dim = kwargs.get("x_dim")
        self.y_dim = kwargs.get("y_dim")
        self.v_num = 100
        self.pos_sim = pos_sim
        self.locations = None

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

    def get_error(self):
        locations = self.get_locations()
        max_local = None
        for local in locations:
            if max_local is None or\
                    local.get_confidence() > max_local.get_confidence():
                max_local = local

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
