
import collections
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import matplotlib.cm as cm
import matplotlib.colors as colors
import rospy


class Drawer(object):

    def __init__(self, x_dim, y_dim):
        self.pub = rospy.Publisher(
            "visualization_marker", Marker, queue_size=1000
        )

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_markers = 1000
        self.duration = 30

        self.clear_all().update()
        self.markers = collections.deque(list(), self.num_markers)
        self.sm = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1))
        self.sm.set_cmap("jet")

    def draw_risk_grid(self, risk_grid):
        if not rospy.is_shutdown():
            p_list = list()
            c_list = list()
            x_gen = xrange(0, self.x_dim, 1)
            y_gen = xrange(0, self.y_dim, 1)

            for i in x_gen:
                for j in y_gen:
                    risk = risk_grid[i, j]
                    pnt = Point(i, j, 0)
                    r, g, b, a = self.sm.to_rgba(risk)
                    clr = ColorRGBA(r, g, b, a)
                    p_list.append(pnt)
                    c_list.append(clr)

            marker = Marker()
            marker.header.frame_id = "/my_frame"
            marker.lifetime = rospy.Duration(10000000)
            marker.type = marker.POINTS
            marker.scale.x = 1
            marker.scale.y = 1
            marker.action = marker.ADD
            marker.points = p_list
            marker.colors = c_list
            marker.id = -1
            self.pub.publish(marker)
        return self

    def draw_nodes(self, positions):
        for i, position in enumerate(positions):
            if not rospy.is_shutdown():
                marker = Marker()
                marker.header.frame_id = "/my_frame"
                marker.lifetime = rospy.Duration(self.duration)
                marker.type = marker.CYLINDER
                marker.action = marker.ADD
                marker.scale.x = 2
                marker.scale.y = 2
                marker.scale.z = 1
                marker.color.a = 1
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.pose.orientation.w = 1
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.position.x = position.x
                marker.pose.position.y = position.y
                marker.pose.position.z = 0
                marker.id = i
                self.markers.append(marker)
        return self

    def draw_source(self, position, color):
        if not rospy.is_shutdown():
            marker = Marker()
            marker.header.frame_id = "/my_frame"
            marker.lifetime = rospy.Duration(self.duration)
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.scale.x = 2
            marker.scale.y = 2
            marker.scale.z = 1
            marker.color.a = 1
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.pose.orientation.w = 1
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.position.x = position.x
            marker.pose.position.y = position.y
            marker.pose.position.z = 0
            marker.id = -2
            self.markers.append(marker)

    def update(self):
        for marker in self.markers:
            self.pub.publish(marker)
        return self

    def clear_all(self):
        self.markers = collections.deque(list(), self.num_markers)
        return self
