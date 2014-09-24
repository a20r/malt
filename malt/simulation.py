
import math
import point
import random
import localization
import time
import numpy as np
from localinstance import LocalInstance
from detectionevent import DetectionEvent


class Simulation(object):

    def __init__(self, **kwargs):
        self.node_positions = kwargs.get("node_positions", list())
        self.intensity_noise = kwargs.get("intensity_noise", 0)
        self.position_noise = kwargs.get("position_noise", 0)
        self.confidence_noise = kwargs.get("confidence_noise", 0)
        self.scene_file = kwargs.get("scene", None)
        self.time_noise = kwargs.get("time_noise", 0)
        self.randomly_generate = kwargs.get("randomly_generate", False)
        self.num_nodes = kwargs.get("num_nodes", 0)
        self.x_dim = kwargs.get("x_dim", 100)
        self.y_dim = kwargs.get("y_dim", 100)
        self.kwargs = kwargs
        self.init_nodes()
        self.init_scene()

    def init_nodes(self):
        if self.randomly_generate:
            for _ in xrange(self.num_nodes):
                n_pos = point.get_random_point(self.x_dim, self.y_dim)
                self.node_positions.append(n_pos)

    def init_scene(self):
        if not self.scene_file is None:
            self.scene = np.loadtxt(self.scene_file)
        else:
            self.scene = np.zeros((self.y_dim, self.x_dim))

    def get_area_noise(self, x, y):
        return 0.2 * self.scene[y, x]

    def provide_stimulus(self, x, y, r_ref, l_ref):
        pos = point.Point(x, y)
        node_events = list()
        for node_pos in self.node_positions:
            dist = pos.dist_to(node_pos)

            intensity_noise = self.get_area_noise(node_pos.x, node_pos.y)
            # intensity_noise = random.gauss(0, self.intensity_noise)

            intensity = localization.intensity_at_distance(
                r_ref, l_ref, dist
            ) + intensity_noise
            node_intensity = intensity_noise * intensity + intensity

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

        return LocalInstance(r_ref, l_ref, pos, node_events, self)

    def step(self, x, y, r_ref, l_ref):
        local = self.provide_stimulus(x, y, r_ref, l_ref)
        max_location = local.get_max_location()
        for i, node_event in enumerate(local.node_events):
            r_dist = max_location.get_position().dist_to(
                node_event.get_position()
            )
            e_dist = localization.distance_from_source(
                r_ref, l_ref, node_event.get_intensity()
            )
            err = abs(r_dist - e_dist)
            r_dir = random.random() * 2 * math.pi
            vel = point.Point(
                err * math.cos(r_dir),
                err * math.sin(r_dir)
            )

            self.node_positions[i] = self.node_positions[i] + vel

        return local
