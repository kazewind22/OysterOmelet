# -*- coding: utf-8 -*-
'''
This is the file containing general functions that are used throughout the
fleetpath library.
However, each section pertains to a specific type of data structure or part
of the problem. Hence if the library grows, these parts may be split off to
different files.

Overivew:
    0) Imports from different python libraries
    1) Network related classes and methods

'''

'''
The first part of the file (which can be later broken off to a separate file
if it grow too large) defines the classes and methods pertaining to networks.
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import collections
import json
from datetime import datetime
import time
from pandas import DataFrame
from datetime import timedelta


def index_node(list_nodes, node_name):
    node_name_list = [item[0] for item in list_nodes]
    # The index() method searches an element in the list and returns
    # its index:
    index_node = node_name_list.index(node_name)
    return index_node

def get_vehicle(vehicle_name, vehicle_config):
    '''
    GET_VEHICLE() is a function that retrieve all the vehicle data from the json
    file and restitutes the dictionnary (or data) of that vehicle.
    '''
    vehicle_name_list = [item['pavId'] for item in vehicle_config]
    # The index() method searches an element in the list and returns
    # its index:
    index_vehicle = vehicle_name_list.index(vehicle_name)

    # Fetches the right element in the vehicle_config list of dictionnaries.
    vehicle = vehicle_config[index_vehicle]
    return vehicle

def index_edge(list_edges, edge):
    list_edges = [sorted(item) for item in list_edges]
    # The index() method searches an element in the list and returns
    # its index:
    index_edge = list_edges.index(sorted(edge))

    return index_edge

def plot_json(jsonFilename):
    '''
    This function plots json files in the terminal of a local json file
    '''
    with open(jsonFilename) as data_file:
        data = json.load(data_file)
    dF = DataFrame(data)
    print(dF)

def vertical_spacing(n_gates, n_FATOs, delta, h_max):
    if n_gates+n_FATOs > h_max/delta:
        delta = h_max/(n_gates+n_FATOs)
    return delta

def find_missing(lst):
    return [x for x in range(lst[0], lst[-1]+1) if x not in lst]

def find_min_max_dates(Schedule, params):
    '''
    This function returns the minimum and maximum values of the times/dates
    in a schedule.
    This function should be removed as soon as the Pandas is used for these purposes.
    '''
    timeFormat = params['timeFormat']

    seq = [x['sTOT'] for x in Schedule]
    min_date = datetime.strptime(min(seq), timeFormat) - params['plot_axis_time_buffer']

    seq = [x['sLDT'] for x in Schedule]
    max_date = datetime.strptime(max(seq), timeFormat) + params['plot_axis_time_buffer']

    return [min_date, max_date]

def avg_time(times):
    '''
    This function is designed to calculate the average date between 2 dates.
    It is used in plotting values (like speed) on the schedule.
    We assume that the average is calclated only within hours (that all times
    up to the day are equivalent, so that the average muct not be compted over
    the entire date)
    '''
    avg = 0
    baseTime = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
    for elem in times:
        avg += elem.second + 60*elem.minute + 3600*elem.hour
    avg /= len(times)
    return baseTime + timedelta(seconds=avg)

def extract_from_trajectory(trajectory, idx_element):
    '''
    This function extracts a given element from a trajectory, such as departure
    vertiport, speed, etc. and returns it as a list.
    We use list comprehension for this:
    '''
    return [item[idx_element] for item in trajectory]

def extract_number_flights(schedule):
    n_flights = 0

    for key in schedule:
        n_flights += len(schedule[key])

    return n_flights

def x_to_step(x,y):
    '''
    This function turns x,y vectors for plotting into a vectors that will plot
    as steps by adding intermediate values to make the steps instead of the
    strait lines.
    '''

    if len(x) != len(y):
        raise Exception('in the x_to_step function the x,y vectors are not the same length. len(x)= {}, len(y)={}'.format(len(x),len(y)))

    x_step = []
    y_step = []
    idx = 0
    for x_i, y_i in zip(x,y):
        idx += 1
        if idx < len(x):
            x_step.append(x_i)
            x_step.append(x[idx])# iterator is one step ahead, thus the element selected here is x_i+1
            y_step.append(y_i)
            y_step.append(y_i)

    return x_step, y_step
