
from point import Point
from location import Location

import math
import scipy.optimize as opt
import sklearn.cluster as clustering  # AffinityPropagation
import numpy as np


STD_SCALE = 30
MIN_DIST = 5  # minimum distance between sources

# Special distance uncertainty threshold in meters
# Used for node association
DISTANCE_THRESHOLD = 2


def distance_from_source(r_ref, l_ref, l_current):
    return r_ref * math.pow(10, abs(l_ref - l_current) / 20.0)


def intensity_at_distance(r_ref, l_ref, r_des):
    return l_ref - abs(20.0 * math.log10(r_ref / r_des))


def distance_from_detection_event(x, y, node_event):
    return Point(x, y).dist_to(node_event.get_position())


def normal_distribution(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * math.pow(x, 2))


def set_node_events_std(node_events):
    """

    Uses the relative time differences between the node events mixed with
    the confidence of source recognition to determine the standard deviation
    of the node event. Node events that are closest to the node event that has
    the largest time stamps will have a lower standard deviation. Also, node
    events that have a larger source recognition confidence will be given lower
    standard deviations.

    @param nodeEvents The list of associated data when a node detects with some
    confidence that the source has been identified

    """

    if len(node_events) == 0:
        raise ValueError("Node event list is of length 0")

    max_time = 0
    min_time = node_events[0].get_timestamp()
    for node_event in node_events:
        if node_event.get_timestamp() > max_time:
            max_time = node_event.get_timestamp()
        elif node_event.get_timestamp() < min_time:
            min_time = node_event.get_timestamp()

    for node_event in node_events:
        time_error = 1.0 - (
            (max_time - node_event.get_timestamp()) /
            (max_time - min_time + 1)
        )
        node_event.set_std(STD_SCALE / (node_event.confidence + time_error))


def position_probability(x, y, r_ref, l_ref, node_events):

    set_node_events_std(node_events)
    pos_eval = 0.0

    for n in node_events:
        dist = distance_from_detection_event(x, y, n)\
            - distance_from_source(r_ref, l_ref, n.intensity)
        norm_val = normal_distribution(dist / n.get_std()) / n.get_std()
        pos_eval += norm_val

    return pos_eval


def determine_source_position_list(r_ref, l_ref, node_events, **kwargs):
    """
    Determines a list of possible positions of where the source will be
    located. These positions are determined by changing iterating through
    the list of node events and optimizing the probability density function
    with the initial guess being the position of the node event. *Hopefully*
    the optimization will find the local minima.

    """

    p_func = lambda v: -1 * position_probability(
        v[0], v[1], r_ref, l_ref, node_events
    )

    max_list = [
        opt.fmin(p_func, ne.get_pos(), full_output=1, **kwargs)
        for ne in node_events
    ]

    max_vals = list()

    for (x, y), z, _, _, _ in max_list:
        max_vals.append((Point(x, y), -z))

    return max_vals


def determine_peaks(opt_vals, label_list):
    """

    Given a list of "optimized" points and their corresponding probabilities
    and a list of labels returned from the clustering algorithm, this
    function goes through all of the optimized points and returns the ones with
    the highest probabilities and issues them as cluster centers. This function
    is used to ensure that the center for each cluster also has the highest
    probability in the cluster of being the source position.

    @param opt_vals A list of tuples where each element is a key-value pair
    where the key is the Point object of the optimization x and y position
    and the value is the associated probability

    @param label_list A list of integers where the index of the list
    corresponds to a certain key-value pair in the first parameter
    list and the integer value represents which cluster it belongs to.

    """

    max_prob_list = list()
    max_point_list = list()
    for i, (point, prob) in zip(label_list, opt_vals):
        try:
            if max_prob_list[i] < prob:
                max_point_list[i] = point
                max_prob_list[i] = prob
        except IndexError:
            max_point_list.append(point)
            max_prob_list.append(prob)

    ret_list = list()
    for max_point in max_point_list:
        too_close = False
        for ret_point in ret_list:
            if ret_point.dist_to(max_point) < MIN_DIST:
                too_close = True
                break
        if not too_close:
            ret_list.append(max_point)

    return ret_list


def determine_source_locations_instance(r_ref, l_ref, node_events, **kwargs):
    """

    Determines the position in the probability grid that has the highest
    probability of being the position of the source.

    """

    max_vals = determine_source_position_list(
        r_ref, l_ref,
        node_events,
        **kwargs
    )

    positions = np.array([p.to_list() for p, _ in max_vals])
    af = clustering.AffinityPropagation().fit(positions)
    max_prob_centers = determine_peaks(max_vals, af.labels_)

    prob_list = [
        position_probability(
            p.x, p.y, r_ref, l_ref,
            node_events
        ) for p in max_prob_centers
    ]

    ret_list = [
        Location(p, conf)
        for p, conf in zip(max_prob_centers, prob_list)
    ]

    return ret_list


def evaluate_location_list(location_list):

    if location_list is None:
        return 0

    locations_conf = 0
    for location in location_list:
        locations_conf += location.get_confidence()

    return locations_conf


def determine_reference_data(r_ref, l_ref, node_events, **kwargs):

    pos_func = lambda ref: -1 * evaluate_location_list(
        determine_source_locations_instance(
            ref[0], ref[1],
            node_events,
            **kwargs
        )
    )

    opt_output = opt.fmin(pos_func, [r_ref, l_ref], full_output=1, **kwargs)
    r_opt, l_opt = opt_output[0]

    return r_opt, l_opt


def get_node_distance_lists(r_ref, l_ref, node_events, locations):

    distance_lists = list()

    for location in locations:
        distance_list = list()
        for node_event in node_events:
            actual_distance = distance_from_detection_event(
                location.x,
                location.y,
                node_event
            )

            predicted_distance = distance_from_source(
                r_ref, l_ref,
                node_event.get_intensity()
            )

            distance_list.append(abs(predicted_distance - actual_distance))

        distance_lists.append(distance_list)

    return distance_lists


def associate_node_events(r_ref, l_ref, node_events, locations):
    """

    Checks with node events correspond to which peaks so we can optimize
    the reference intensity and reference distance. If we
    know the node event association, we can partition the problem into
    multiple sets of node events for multiple peaks. Then using this
    partitioning, we can optimize different values or r_ref and l_ref for
    different source occurences

    """

    distance_lists = get_node_distance_lists(
        r_ref, l_ref,
        node_events,
        locations
    )

    association_dict = dict()

    for location_index, distance_list in enumerate(distance_lists):
        for node_index, distance in enumerate(distance_list):
            if distance < DISTANCE_THRESHOLD:
                if not location_index in association_dict.keys():
                    association_dict[locations[location_index]] = list()
                association_dict[locations[location_index]].append(
                    node_events[node_index]
                )

    return association_dict


def determine_source_locations(r_ref, l_ref, node_events, **kwargs):

    initial_source_locations = determine_source_locations_instance(
        r_ref, l_ref,
        node_events,
        **kwargs
    )

    node_event_associations = associate_node_events(
        r_ref, l_ref,
        node_events,
        initial_source_locations
    )

    location_list = list()

    for event_list in node_event_associations.values():
        r_opt, l_opt = determine_reference_data(
            r_ref, l_ref,
            event_list,
            **kwargs
        )

        location_list += determine_source_locations_instance(
            r_opt, l_opt,
            event_list,
            **kwargs
        )

    return location_list
