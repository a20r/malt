
import math
import point
import random
import localization
import time
import numpy as np
import rospy
from localinstance import LocalInstance
from detectionevent import DetectionEvent


class Simulation(object):

    def __init__(self, **kwargs):
        rospy.init_node("malt", anonymous=False)
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
        self.starting_radius = 10
        self.kwargs = kwargs
        self.num_iter = 1
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
        return 0.3 * self.scene[y, x]

    def provide_stimulus(self, x, y, r_ref, l_ref):
        pos = point.Point(x, y)
        node_events = list()
        for node_pos in self.node_positions:
            dist = pos.dist_to(node_pos)

            intensity_noise = self.get_area_noise(node_pos.x, node_pos.y)
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

    def determine_velocity(self, local, node_event, rad=100):
        neighbours = self.get_node_neighbours(
            node_event, local.node_events, rad
        )
        err = self.get_node_error(local, node_event)

        # if self.num_iter < 50:
        #     next_x = 0.1 * err * math.cos(1 * self.num_iter)
        #     next_y = 0.1 * err * math.sin(1 * self.num_iter)
        #     next_pos = point.Point(next_x, next_y)
        #     return next_pos

        if len(neighbours) > 0:
            n_weights, w_sum = self.get_neighbour_weights(
                local, node_event, neighbours
            )
            direction = point.Point(0, 0)

            for gnn, weight in zip(neighbours, n_weights):
                p = point.Point(gnn.x - node_event.x, gnn.y - node_event.y)
                p = p.to_unit_vector()
                direction = direction + p * (weight / w_sum)

            vel = direction * err * 0.2
            return vel
        else:
            return self.get_random_velocity() * 0.2 * err

    def get_random_velocity(self):
        theta = random.random() * 2 * math.pi
        return point.Point(math.cos(theta), math.sin(theta))

    def get_neighbour_weights(self, local, node_event, neighbour_events):
        n_weights = list()
        sum_weights = 0.0
        for n_ev in neighbour_events:
            # dist = node_event.get_position().dist_to(n_ev.get_position())
            err = self.get_node_error(local, n_ev)
            weight = 1 / (err)
            sum_weights += weight
            n_weights.append(weight)
        weights = [w if w == max(n_weights) else 0 for w in n_weights]
        return weights, sum(weights)

    def get_node_neighbours(self, n_event, node_events, search_radius):
        neighbours = list()
        for n_ev in node_events:
            dist = n_event.get_position().dist_to(n_ev.get_position())
            if dist < search_radius and dist > 2:
                neighbours.append(n_ev)
        return neighbours

    def get_node_error(self, local, node_event):
        max_location = local.get_max_location()
        r_dist = max_location.get_position().dist_to(
            node_event.get_position()
        )
        e_dist = localization.distance_from_source(
            local.r_ref, local.l_ref, node_event.get_intensity()
        )
        err = abs(r_dist - e_dist)
        return err

    def step(self, x, y, r_ref, l_ref):
        self.num_iter += 1
        local = self.provide_stimulus(x, y, r_ref, l_ref)
        for i, node_event in enumerate(local.node_events):
            vel = self.determine_velocity(local, node_event)
            self.node_positions[i] = self.node_positions[i] + vel
        return local
