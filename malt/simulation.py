
import point
import random
import localization


class Simulation(object):

    def __init__(self, *nodes, **kwargs):
        self.nodes = nodes
        self.intensity_noise = kwargs.get("intensity_noise", 0)
        self.position_noise = kwargs.get("position_noise", 0)

    def provide_stimulus(self, x, y, intensity):
        pos = point.Point(x, y)
        for node in self.nodes:
            dist = pos.dist_to(node.get_position())
            node_intensity = self.get_intensity_from_distance(dist)
            noise = random.gauss(0, self.intensity_noise)
            node.provide_stimulus(node_intensity + noise)

        return self

    def localize(self):
        for node in self.nodes:
            pass
