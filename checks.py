# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import collections
import json
from datetime import datetime
import time
from pandas import DataFrame
from network import network
import utility_functions as uF


def check_nodes_in_schedule():
    '''
    The goal of the check_nodes_in_schedule function is to verify that all the
    nodes within the schedule are also in the network. If this is not the case,
    the software may have bugs further down the line.
    '''
    return 0

def check_speed(schedule, Network, vehicle_config):

    v_schedule = {}
    flight_labels = {}

    for trajectory_name in schedule:

        trajectory      = schedule[trajectory_name]
        trajectory_new  = []
        labels          = []

        for flight in trajectory:
            t0              = flight[0]
            x0_name         = flight[1]
            t1              = flight[2]
            x1_name         = flight[3]
            vehicle_name    = flight[5]

            vehicle         = uF.get_vehicle(vehicle_name, vehicle_config)
            V_max           = vehicle['V_max']
            V_min           = vehicle['V_min']

            (d_N1toN2, connectivity) = Network.get_distance(x0_name, x1_name) # calculation of the distance separating the departure and arrival

            # if the flight distance is 0, then the vehicle is waiting, so a
            # special condition is created where waiting does not violate the
            # speed limits. An artificial speed if set so that the speed limit
            # condition is not triggered.


            if d_N1toN2 == 0:
                V = 0.5*(V_max + V_min)
            else :
                t0 = time.mktime(t0.timetuple())
                t1 = time.mktime(t1.timetuple())
                deltaT = t1-t0
                epsilon = 10e-6 # an epsilon is added in order to avoid division by zero should that occur.
                V = d_N1toN2/(deltaT+epsilon) # calculation of the vehicle speed

            if V > V_max:
                # The speed is added to the trajectory:
                flight[7] = V
                trajectory_new.append(flight)
                print('too fast:')
                print(V)
                print('distance [km]')
                print(d_N1toN2/1000)

            if V< V_min:
                # The speed is added to the trajectory:
                flight[7] = V
                trajectory_new.append(flight)
                print('too slow:')
                print(V)
                print('distance [km]')
                print(d_N1toN2/1000)

        v_schedule[trajectory_name] = trajectory_new

    return v_schedule

def check_interference(completedSchedule, Network, vehicle_config):
    '''
    The goal of this function is to check for interference between the different
    trajectories on vertiports and gates.
    Functioning: it skims through a completed_schedule, and only returns the
    parts of the schedule where there is a conflict.
    '''
    interference_schedule = {}

    for trajectory_name in completedSchedule:

        trajectory = completedSchedule[trajectory_name]

        new_trajectory = []

        for flight in trajectory:

            if len(flight)>6: # if there is a vehicle_parameter dictionary:

                if 'conflict' in flight[6]: # if the is a key 'conflict' in the vehicle_parameter dictionnary

                       new_trajectory.append(flight)

        interference_schedule[trajectory_name] = new_trajectory


    return interference_schedule

def check_connectivity():
    connectivity_schedule = 0
    return connectivity_schedule

def check_vehicle_capacity(InitialSchedule, Network, vehicle_config):
    '''
    The goal of this function is to check that the schedule is not overfilling
    or underfilling the vehicles according to their capacity. It updates the
    schedule by specifying the excess number of passengers that the vehicle
    can carry. For example, if the vehicle is exactly full, return 0 on the
    leg. If there is one space left, return 1. If there are 2 people too many
    on the vehicle (4 people instead of 2), return -2.
    Note: do not use an completed schedule in this case. An initial schedule
    is far better since it only has the flight legs. Passenger capacity should
    be done on the initial schedule.
    '''
    capacity_schedule = {}

    for trajectory_name in InitialSchedule:

        trajectory = InitialSchedule[trajectory_name]

        new_trajectory = []

        for flight in trajectory:

            Npax            = flight[4]
            vehicle_name    = flight[5]

            vehicle         = uF.get_vehicle(vehicle_name, vehicle_config)
            capacity        = vehicle['Pax_max']

            excess_capacity = capacity - int(Npax)

            if len(flight)>6: # if there is a vehicle_parameter dictionary, complete it.

                flight[6]['excess_capacity']      = excess_capacity
            else: # if there is no dictionnary, add it with the appropriate field.

                vehicle_states = {}
                vehicle_states['excess_capacity'] = excess_capacity
                flight[6] = vehicle_states

            new_trajectory.append(flight)


        capacity_schedule[trajectory_name] = new_trajectory


    return capacity_schedule




