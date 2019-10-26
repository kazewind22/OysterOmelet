# -*- coding: utf-8 -*-
'''
This is the file containing all the FleetPath classes and functions. Since the
tool is currently simple, there is no need to disperse it to different files.
However, each section pertains to a specific type of data structure or part
of the problem. Hence if the library grows, these parts may be split off to
different files.

Overivew:
    0) Imports from different python libraries
    1) x-t diagram related classes and methods
    2) Post-processing-related classes and methods
'''

'''
The first part of the file (which can be later broken off to a separate file
if it grow too large) defines the classes and methods pertaining to networks.
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import collections
from datetime import datetime
import time
import network as nW
import utility_functions as uF
import checks as cH

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.cm as cmx
from matplotlib.dates import DateFormatter

import pandas as pd
#from pandas.plotting import register_matplotlib_converters

#register_matplotlib_converters()

def plot_schedule(Network, schedule, title, params, ax=None, **kwargs):
    ''' Plot schedule is the x-t diagram
    A schedule is made of a dictionnary of vehicles and their trajectory:

        schedule S = {V1:T1, V2:T2, ..., V_i:T_i, ... Vn:Tn}

    Each vehicle trajectory is a list of flights (or trajectory segment, which
    includes aiting at a location), each flight containing the
    time, the location, and other imformation such as amount of passengers on
    the aircraft. Currently, since all vehicle are assumed to fly at constrant
    speed, a trajectory can be defined in the x-t space as a series of
    connected segments. The flights that are written down are only those when
    the vehicle state changes, i.e. at the joint between segments:

     x
        ^                   s6           s7
     c  +- - - - - - - - - - -------------
        |    s2  s3         /
     b  +- - ----- - - - - / - - - - - - -
        |   /     \       /
     a  +--- - - - ------- - - - - - - - -
      s0|  s1     s4    s5
        +------------------------------------------> t
        0  1   2   3    4    5    6    7   8
        The trajectory above would be represented by the following vector:
        T1 = [f0, f1, f2, f3, f4, f5, f6, f6]

        And each flight has the following compostion:

            f1 = [t_dep, x_dep, t_arr, x_arr, N_pax, PAV_Id, vehicle_states]

    Notes: the vehicle_states is a dictionnary that contains extra information
    of the trajecotry component. It may be completed as processing goes on.

    Plot schedule functions by:
        1) Setting up an x-t figure upon which to draw the trajectories.
        2) Looping through the schedule, and drawing each trajectory.
                3) Each trajecotry is converted to a series of segments.


    '''
    # creates a set of axis that are then used to be subplotted:
    ax = ax or plt.gca()

    node_sequence =  nW.NetworkTo1D(Network)

    ax.set_title(title)

    ax.set_xlabel('Time')
    ax.set_ylabel('Nodes')

    flight_labels = {}
    speed_graph = False
    capacity_graph = False

    for key, value in kwargs.items():
        if key == 'xlim':
            ax.set_xlim(value)
        if key == 'flight_labels':
            flight_labels = value
        if key == 'speed':
            speed_graph = value
        if key == 'capacity':
            capacity_graph = value

    ax.set_yticks(np.arange(len(node_sequence)))
    ax.set_yticklabels(node_sequence)

    jet = plt.get_cmap('jet')
    values = range(len(schedule))
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    colorVal = scalarMap.to_rgba(0)

    ax.yaxis.grid() # horizontal lines

    ''' Loops through the nodes in order to plot the gates and FATOs appropriately.
    Function visualization parameters.
    '''


    # iterator for the position of the node on the y-axis:
    idx = 0
    for node_name in node_sequence:
        # Get the node index:
        node_idx = uF.index_node(Network.list_nodes, node_name)
        # Get the number of FATOs:
        n_FATOs = Network.list_nodes[node_idx][3]
        # Get the number of gates:
        n_gates = Network.list_nodes[node_idx][4]

        # Test if there are more gates/fatos than the standard split:
        # if there are more gates and fatos than can be plotted with
        # the standard delta...
        delta = uF.vertical_spacing(n_gates, n_FATOs, params['delta'], params['h_max'])

        # Plot the FATOs and gates accordingly:
        jdx = 0

        # Plot the FATOS:
        for i in range(n_FATOs):
            ax.axhline(idx- jdx*delta, color='black', lw=2)
            jdx += 1
        # PLot the gates:
        for i in range(n_gates):
            ax.axhline(idx- jdx*delta, color='black', lw=0.5)
            jdx += 1

        idx += 1


    ''' For loop going through the schedule, showing each trajectory:
    The loop goes throught the schedule, which is a dictionnary with the keys
    being the vehicle names and the values being the trajectories (list of flights)

    Each trajectory is converted to a series of (x,y) coordinates that are then
    plotted. The trajectories are plotted according to flight segments. Holes
    inside the flight segments are possible if the schedule is not complete.

    Input schedules only show the flights, whereas complete schedules are
    complete with transit legs and waiting legs.'''

    idx = -1
    patches = []
    for key in schedule:

        lines = trajectory2plot(schedule[key], node_sequence, params)

        idx += 1
        colorVal = scalarMap.to_rgba(values[idx])
        patches.append(mpatches.Patch(color=colorVal, label=key))

        # If we are plotting a speed graph, the speed is plotted on the lines,
        # and the speed must be extracted from the data set:
        if speed_graph== True:
            speeds = uF.extract_from_trajectory(schedule[key], 7)
        if capacity_graph== True:
            vehicleStates = uF.extract_from_trajectory(schedule[key], 6)

        jdx = 0 # iterator along each flight segment
        for line in lines:

            if capacity_graph == True:
                if vehicleStates[jdx]['excess_capacity'] < 0: # condition met is the vehicle is overloaded:
                    ax.plot(line[0], line[1], '--', linewidth=2*params['linewidth'], color = colorVal)

                elif vehicleStates[jdx]['excess_capacity'] > 0: # condition met is the vehicle is underloaded:
                    ax.plot(line[0], line[1], '--', linewidth=0.5*params['linewidth'], color = colorVal)

                else:
                    ax.plot(line[0], line[1], 'o-', linewidth=params['linewidth'], color = colorVal)

            else:

                ax.plot(line[0], line[1], 'o-', linewidth=params['linewidth'], color = colorVal)

            # If we are plotting a speed graph, the speed is plotted on the lines:
            if speed_graph== True:

                list_dates = line[0]
                average_date = uF.avg_time(list_dates)
                ax.text(average_date , np.mean(line[1]), round(speeds[jdx],2), fontsize=params['figFontSize'])


            if capacity_graph== True:
                list_dates = line[0]
                average_date = uF.avg_time(list_dates)
                ax.text(average_date , np.mean(line[1]), vehicleStates[jdx]['excess_capacity'], fontsize=params['figFontSize'])
            jdx += 1

    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')
    ax.legend(handles=patches)

    return ax

def extract_edge_name(pandas_demand):
    '''
    This function is used to get the edge names from a pandas dataframe. Note
    that is is used as a quick fix currently, for the final version should have
    the dates figured out.
    '''
    # Convert the demand to a Pandas dataframe:
    pandas_demand.sort_values(by=['departureVertiport','arriveVertiport'], inplace=True)

    # Select the inital edge. Every time the edge changes, it is stored and we go
    # onto the next.
    previousEdge = [pandas_demand.iloc[0]['departureVertiport'],pandas_demand.iloc[0]['arriveVertiport']]

    labels =[] # declare labels

    idx = 0
    for index, row in pandas_demand.iterrows():
        idx += 1
        # if there is no change, i.e. we are on the same edge:
        if row['arriveVertiport'] == previousEdge[1]:
            # if we have reached the end of the dataframe, there is no change to the next next edge. Thus it must be added:
            if (idx == len(pandas_demand)) & (row['arriveVertiport'] == previousEdge[1]):
                # We append the previous string to the existing lists:
                labels.append(str(previousEdge[0]+ ' - ' + previousEdge[1]))

        # if there is a change in the arrival/departure vertiports, then a new series is made
        else:
            # We append the previous string to the existing lists:
            labels.append(str(previousEdge[0]+ ' - ' + previousEdge[1]))
            previousEdge = [row['departureVertiport'],row['arriveVertiport']]

    return labels


def plot_demand(demand, params, ax=None):
    '''

     Npax
        ^
     c  +                ------
        |                |    |      ------
     b  +    -------     |    |      |
        |    |     |     |     -------
     a  +-----     -------
        |
        +------------------------------------------> t
        0    1     2     3    4     5    6    7   8

        Demand profile. There are N pax per time interval.
    '''
    # creates a set of axis that are then used to be subplotted:

    title   = 'demand profile '
    ax      = ax or plt.gca()
    ax.set_title(title)

    ax.set_xlabel('Time')
    ax.set_ylabel('Pax demand')


    ax.yaxis.grid() # horizontal lines

    # Convert the demand to a Pandas dataframe:
    demand_df = pd.DataFrame.from_dict(demand, orient='columns')
    demand_df.sort_values(by=['departureVertiport','arriveVertiport'], inplace=True)

    X = []
    Y = []

    # Select the inital edge. Every time the edge changes, it is stored and we go
    # onto the next.
    previousEdge = [demand_df.iloc[0]['departureVertiport'],demand_df.iloc[0]['arriveVertiport']]
    x = []
    y = []
    landingTimes = []

    labels =[] # declare labels

    idx = 0
    for index, row in demand_df.iterrows():
        idx += 1
        # if there is no change, i.e. we are on the same edge:
        if (row['arriveVertiport'] == previousEdge[1]) & (row['departureVertiport'] == previousEdge[0]):

            #x.append([row['sTOT'], row['sLDT']])
            x.append(row['sTOT'])

            # Npax is put in twice because we want to have "zero order hold"
            # graph (need for Npax at beginning of segment and at end.
            # In this case both are the same).
            # We also have to cast the number of passengers into floats
            # for plotting:

            #y.append([float(row['Npax']), float(row['Npax'])])
            y.append(float(row['Npax']))

            landingTimes.append(row['sLDT'])

            # if we have reached the end of the dataframe, there is no change to the next next edge. Thus it must be added:
            if idx == len(demand_df):
                # We want to add the collected lists. Before this is done they have to
                # be sorted:
                y_sorted = [y for _,y in sorted(zip(x,y))]
                x_sorted = sorted(x)
                x_sorted.append(max(landingTimes))
                y_sorted.append(y_sorted[-1])
                X.append(x_sorted)
                Y.append(y_sorted)

                # We append the previous string to the existing lists:
                labels.append(str(previousEdge[0]+ ' - ' + previousEdge[1]))

        # if there is a change in the arrival/departure vertiports, then a new series is made
        else:
            # We want to add the collected lists. Before this is done they have to
            # be sorted:

            y_sorted = [y for _,y in sorted(zip(x,y))]
            x_sorted = sorted(x)
            x_sorted.append(max(landingTimes))
            y_sorted.append(y_sorted[-1])
            X.append(x_sorted)
            Y.append(y_sorted)

            # We append the previous string to the existing lists:
            labels.append(str(previousEdge[0]+ ' - ' + previousEdge[1]))
            previousEdge = [row['departureVertiport'],row['arriveVertiport']]
            # Initialize empty lists for the next round:
            x = []
            y = []
            x.append(row['sTOT'])
            y.append(float(row['Npax']))
            landingTimes.append(row['sLDT'])

    patches = []

    jet = plt.get_cmap('jet')
    values = range(len(labels))
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    colorVal = scalarMap.to_rgba(0)

    # Fix: currently the date do not plot correctly. Thus the demand profiles are plotted one next to another for the time being.
    # So we specify an edge to plot in the function...

# TO BE USED ONCE PLOTTING DATES IS SOLVED:
    idx = -1
    for x,y in zip(X,Y):
        idx += 1

        x = X[idx]
        y = Y[idx]

        # For plotting steps using the plot, the coordinates need to be duplicated accordingly.:
        (x_step, y_step) = uF.x_to_step(x,y)

        colorVal = scalarMap.to_rgba(values[idx])
        patches.append(mpatches.Patch(color=colorVal, label=labels[idx]))
        # Need to cast y into a float:
        ax.plot(x_step, y_step, 'o-', linewidth=params['linewidth'], color = colorVal)

    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')
    ax.legend(handles=patches)

    return ax

def trajectory2plot(trajectory, node_sequence, params):
    ''' TRAJECTORY2PLOT is a function which turns a trajectory, as defined in
    the PLOT_SCHEDULE function, into a series of X,Y coordinates that will be
    used to plot the schedule. The node sequence is used to map the names of
    the nodes to a number, which corresponds to the y-axis value on an x-t
    diagram.
    In this case, the time values t_i can be directly mapped to x values (both)
    are continuous.

    Currently, this task is solved by a loop over all time stamps:
    '''
    X = []
    Y = []

    lines = []

    for flight in trajectory:

        # The x-coordinate is the date (time). The departure time (flight[0])
        # and arrival time (flight[2]) is extracted from the individual flight:
        x0 = flight[0]
        x1 = flight[2]


        # Adding the decrements for the Y coordinate to plot the different FATOs
        # and gate operations:
        # First the decrement in the first location:
        if len(flight) < 7: # if the length of the flight is less than 6, then the flight states dictionary does not exist,thus there is not decrement.
            delta_1 = 0
        elif 'delta_1' in flight[6]:
            delta_1 = flight[6]['delta_1']
        else:
            delta_1 = 0

        if len(flight) < 7: # if the length of the flight is less than 6, then the flight states dictionary does not exist,thus there is not decrement.
            delta_2 = 0
        elif 'delta_2' in flight[6]:
            delta_2 = flight[6]['delta_2']
        else:
            delta_2 = 0

        y0 = uF.index_node(node_sequence, flight[1])+delta_1
        y1 = uF.index_node(node_sequence, flight[3])+delta_2

        lines.append([[x0, x1],[y0,y1]])

    return lines


def json2schedule(Schedule, vehicle_config_file_name, Network, params):
    '''
    This functions turns a json schedule into a 'stick' schedule made of
    a series of trajectories, each made of a series of states (see plot_schedule).

    Assuming the json file is in the same directory as the fleetpath code

    Function structure: (loop in loop... Should be improved.)

            for  flightSegment in json_file:

                get_vehicle_id:
                    if new vehicle, make new trajectory

                    else add flight to the trajectory vector according to time stamp


            sort the trajectories according to the timestamp

    INPUTS:
        json_schedule_file_name: the is the string containing the name of the
                                 schedule in json format. Example:
                                     "json_schedule_test_normal.json"
        json_vehicle_config_file_name: the is the string containing the name of the
                             vehicle configuration file in json format. Example
                             is "json_vehicle_config.json"
        Network is the name of a network class instance, as defined above.

        params is a dictionnary that contains parameters pertaining to the function.
                            in this case it is used for the time format, which
                            can vary from one input file to the other potentially.

    OUTPUTS:
        The schedule, as in a "stick" schedule that may be used for plotting
        and geometric verification.

    '''

    '''            schedule S = {V1:T1, V2:T2, ..., V_i:T_i, ... Vn:Tn}

    Each vehicle trajectory is a list of states, each state containing the
    time, the location, and other imformation such as amount of passengers on
    the aircraft. Currently, since all vehicle are assumed to fly at constrant
    speed, a trajectory can be defined in the x-t space as a series of
    connected segments. The states that are written down are only those when
    the vehicle state changes, i.e. at the joint between segments:

     x
        ^                   s6           s7
     c  +- - - - - - - - - - -------------
        |    s2  s3         /
     b  +- - ----- - - - - / - - - - - - -
        |   /     \       /
     a  +--- - - - ------- - - - - - - - -
      s0|  s1     s4    s5
        +------------------------------------------> t
        0  1   2   3    4    5    6    7   8
        The trajectory above would be represented by the following vector:
        T1 = [f0, f1, f2, f3, f4, f5, f6, f6]

        And each flight has the following compostion:

        f1 = [t_dep, x_dep, t_arr, x_arr, N_pax, PAV_Id, vehicle_states]

    Notes: the vehicle_states is a dictionnary that contains extra information
    of the trajecotry component. It may be completed as processing goes on.

               '''
    timeFormat = params['timeFormat']
    schedule = {}
    vehicle_names = [] # list of vehicles/trajecotries in the network.

    # First step: separation of all the flights into trajectories:

    for flight_segment in Schedule:

        # create a 'state' as defined above:

        # Note: we assume in this code version (9/26/2019) that FATO and
        # vertiport are interchangeable.

        x0      = flight_segment['departureVertiport']
        x1      = flight_segment['arriveVertiport']
        # Note: the current json file format does not include the PAX number
        # A new field is added to the current example file.
        Npax    = flight_segment['Npax']

        # Extract the times, convert them and calculate the difference:
        s0 = flight_segment['sTOT']
        s1 = flight_segment['sLDT']
        t0 = datetime.strptime(s0, timeFormat) # timeFormat is a string that sets the format in which the time is stored, such as "%Y-%m-%dT%H:%M:%SZ"
        t1 = datetime.strptime(s1, timeFormat)

        #t0 = time.mktime(t0.timetuple())
        #t1 = time.mktime(t1.timetuple())

        # Verify the flight speed constraint:
        vehicle_name    = flight_segment['pavId']

        vehicle_states = {}

        speed_vehicle = 0

        flight = [t0, x0, t1, x1, Npax, vehicle_name, vehicle_states, speed_vehicle]
        # Note: s0 does not have to be the first flight of the segment.
        # the states are ordered in the subsequent steps.

        # Note: because of the definition, a flight represents 2 states as
        # opposed to a single state

        if flight_segment['pavId'] in vehicle_names:

            # initiate the trajectory with a first state.
            schedule[flight_segment['pavId']].append(flight)

        # If there are no vehicles of such name in the list yet, add
        else:
            # update the vehicle_names vector:
            vehicle_names.append(flight_segment['pavId'])

            # initiate the trajectory with a first state.
            schedule[flight_segment['pavId']] = [flight]

    # Second step: verification of the trajectory continuity
    '''
    The continuity of the trajectory is proven recursively (segment by segment)
    3 conditions have to be met to have trajecotry continuity in this case:
        1. once the algorithm has sorted all segments, there must be none left
        (i.e. no free standing segments)
        2. The arrival location of the previous segment must be the departure
        location of the next segment
        3. The departure time of the next segment must be after the arrival time
        of the previous segment.
    '''

    for trajectory in schedule:

        # Order the segments by departure time:

        schedule[trajectory] = sorted(schedule[trajectory])

    return schedule

def jsonToPandas(json_data):
    '''
    Switches the json schedule format to a
    pandas dataframe showing flights.
    '''
    panda_schedule = pd.DataFrame.from_dict(json_data, orient='columns')
    return panda_schedule

def check_demand_capacity(panda_Schedule, Network, vehicle_config, demand, params):
    '''
    This check verifies that the schedule in question is not 'making' passengers
    fly routes they do not intend or create passengers ex-nihilo from the
    vertiports.
    The graph produced shows a demand graph where negative values indicate that
    passengers have been 'created' by the schedule.
    '''
    k = panda_Schedule.drop(['flightId', 'arriveFato', 'departureFato', 'missionId', 'pavId'], axis=1)
    k['absRel'] = pd.Series(1, index=k.index)
    demand['absRel'] = pd.Series(0, index=demand.index)

    remaining_demand = pd.concat([demand, k], ignore_index=True)

    remaining_demand.sort_values(by=['departureVertiport','arriveVertiport', 'sTOT'], inplace=True)

    new_dF = remaining_demand.copy()

    idx = 0 # separate iterator.
    previous_arriveVertiport = remaining_demand.iloc[0]['arriveVertiport']
    for index, row in remaining_demand.iterrows():

        if row['absRel'] == 1: # if the row corresponds to a flight, then the value is relative and it is subtracted to the demand.
            if remaining_demand.iloc[idx]['arriveVertiport'] != previous_arriveVertiport: # if it is the first row for the given leg, we initiate with zero.

                new_dF.at[index,'Npax'] = -int(remaining_demand.iloc[idx]['Npax'])

            else:
                new_dF.at[index, 'Npax'] = int(new_dF.iloc[idx-1]['Npax']) - int(remaining_demand.iloc[idx]['Npax'])

        else: # if the row corresponds to a demand, then this is automatically added to the absolute value on that edge:

              new_dF.at[index,'Npax'] = int(remaining_demand.iloc[idx]['Npax'])

        previous_arriveVertiport = remaining_demand.iloc[idx]['arriveVertiport']
        idx += 1


    new_dF.reset_index(drop= True)

    # Clean the data set by making sure that all entries have a different time (this then used for plotting.)
    # Note that grouping the instances would be a bad idea since we would loose information.
    # Thus a slight offset is given (~1 second) to every instance that has the exact same
    # value:

    idx = 0
    for index, row in remaining_demand.iterrows():
        idx +=1
        if idx < len(remaining_demand):
            if remaining_demand.iloc[idx]['sTOT'] == row['sTOT']: # if the two times are exactly equal, then change the one in question:
                new_dF.at[index,'sTOT'] = remaining_demand.iloc[idx-1]['sTOT'] - params['time_epsilon']

    return new_dF

def check_continuity(schedule):
    '''
    This function verifies the continuity of the schedule, i.e. makes sure that
    the schedule does not have vehilces which are discontinuous in time or
    in space.
    Note that the time constraint is that the vehicle always move forward in
    time.
    INPUTS:
        schedule: the schedule is that output by the json2schedule function.

    OUTPUTS:
        schedule: returns a list of trajectory segments (i.e. flights or waiting
        times) that break the continuity of the schedule. This can be plotted
        on top of an existing schedule to show where the discontinuities lie.

    The algorithm used here is a tracking algorithm, that loops through each
    trajectory, and in turn loops through each flight in a given trajectory
    and tests each flight against conditions given by the previous flight to
    see whether the trajecotry is continuous w.r.t. the previous one. If the algorithm
    can go through the entire trajectory without encountering a moment when the
    flights are "discontinuous" to the previous one, then the trajectory is
    said to be continuous.

    The tests performed are described below. Note that they depend on the way
    the data structures are organised. In this the conditions rely on the
    fact that each state is classified by its time-stamp. Thus one does not
    have to verify that time stamps are increasing from one state to the next,
    however the type of state (departure and arrival) must be in alternating
    order AND the locations between arrival and departures must be the same.

    These 2 tests are sufficient to demonstrate continuity of the schedule:

        - If the 'arrival' time is before the 'departure' time, this signifies
        that there is a time inconsistency, i.e. a vehicle departs an airport
        befor it can leave it.
        -and-
        If the next departure time is before the previous arrival time, an
        discontinuity is flagged.

        - If the location is not the same between an 'arrival' state and a
        'departure' state, then the vehicle 'pops up' at a new airport. This
        is a space inconsistency.

    '''

    inconsistent_schedule = {}

    for trajectory_name in schedule:

        # The function returns trajectories that show the discontinuities
        #specifically. They can be plotted on top of existing schedules in
        # order to show where the inconsistency is:
        inconsistent_trajectory = []

        trajectory = schedule[trajectory_name]

        arrival_loc_1   = trajectory[0][1] # the start has by definition no history and thus takes off from its initial location
        arrival_time_1  = trajectory[0][0]

        for flight in trajectory:

            departure_time  = flight[0]
            departure_loc   = flight[1]
            arrival_time    = flight[2]
            arrival_loc     = flight[3]



            if arrival_time < departure_time:

                #print 'The trajectory has a time inconsistency (aircraft arrives before it leaves)'

                # If this is the case, add the trajectory leg to the "inconsistent trajecotry".
                # This is correct since the inconsistency is within the leg:
                inconsistent_trajectory.append(flight)

            elif departure_loc != arrival_loc_1:

                print('The trajectory has a space inconsistency (aircraft departs from a different location than it arrived at)')
                t_dep   = arrival_time_1
                x_dep   = arrival_loc_1
                t_arr   = departure_time
                x_arr   = departure_loc
                N_pax   = 0 # No passengers are supposed to be carried on a leg that is to be a waiting leg.
                PAV_Id  = flight[5]
                vehicle_states = {}
                hold = [t_dep, x_dep, t_arr, x_arr, N_pax, PAV_Id, vehicle_states]
                inconsistent_trajectory.append(hold)

            elif departure_time < arrival_time_1:

                #print 'The trajectory has a time inconsistency (aircraft departs before it arrives)'

                t_dep   = arrival_time_1
                x_dep   = arrival_loc_1
                t_arr   = departure_time
                x_arr   = departure_loc
                N_pax   = 0 # No passengers are supposed to be carried on a leg that is to be a waiting leg.
                PAV_Id  = flight[5]
                vehicle_states = {}
                hold = [t_dep, x_dep, t_arr, x_arr, N_pax, PAV_Id, vehicle_states]

                #print 'The space inconsistency flight that is added to the inconsistent flight:'
                #print hold
                inconsistent_trajectory.append(hold)

            arrival_loc_1   = flight[3]
            arrival_time_1  = flight[2]

        inconsistent_schedule[trajectory_name] = inconsistent_trajectory

    if all(value == [] for value in inconsistent_schedule.values()) == True:
        print('Schedule is continuous')
    else:
        print('the schedule contains discontinuities')

    return inconsistent_schedule


def complete_schedule(Network, schedule, params):
    '''
    This function takes in a continuous schedule, and then completes it by adding
    all the ground operations (FATOs and gates).
    It transforms a straight waiting line to a subschedule that describes the
    operations within the vertiport:

    Before:


  x ^               vehicle i
    |              /
  A |    _________/
    |   /
    |  /
    | /
    +---------------->  t

    After:

  x ^
    |                             vehicle i
    |                             /
    |                            /
    |   FATO arr.    FATO dep.  /
    |     1____2         5 ____/ 6
    |     /    \         /
    |    /      \_______/ gate 2 fato transit
    |   /       3      4
    |  /        Transit time
    | /
    +--------------------------------------> t

    The points 1-> 6 are calculated using the parameters in the parameter file
    such as FATO transit time (t_2-t_1 & t_6-t_5) and the transit time from
    gate to FATO (t_3-t_2 and t_5-t_4)

    INPUTS:
        Network:

        schedule: Schedule is a schedule in standard format that has been
        verified to be continuous.

    '''

    '''
    f1 = [t_dep, x_dep, t_arr, x_arr, N_pax, PAV_Id, vehicle_states]
    We add the fat odata into the vehicle states !!!!
    '''
    node_sequence =  nW.NetworkTo1D(Network)

    # iterator for the position of the node on the y-axis:
    '''
    Data structure containing the n. FATO, n. gate, and delta for node i:
    nFng = [[n_FATO_1, n_gate_1, delta_1],
                    ...                  ,
            [n_FATO_i, n_gate_i, delta_i],
                    ...                  ,
            [n_FATO_N, n_gate_N, delta_N]]
    '''
    nFnG = []
    for node_name in node_sequence:
        # Get the node index:
        node_idx = uF.index_node(Network.list_nodes, node_name)
        # Get the number of FATOs:
        n_FATOs = Network.list_nodes[node_idx][3]
        # Get the number of gates:
        n_gates = Network.list_nodes[node_idx][4]

        # Test if there are more gates/fatos than the standard split:
        # if there are more gates and fatos than can be plotted with
        # the standard delta...
        delta   = uF.vertical_spacing(n_gates, n_FATOs, params['delta'], params['h_max'])
        nFnG.append([n_FATOs, n_gates, delta])


    ''' Creation of the data structures containing the occupancies of FATOs and
    gates.

    FATO_occupancy = [  FATO_node_1_occupancy,
                        ...
                        FATO_node_i_occupancy,
                        ...
                        FATO_node_N_occupancy]

    Data structure containing the FATO occupancy for node i:
    FATO_node_i_occupancy =  [[FATO_idx, start_time, end_time],  % occupancy 1
                                  ...                  ,
                              [FATO_idx, start_time, end_time],  % occupancy i
                                  ...                  ,
                              [FATO_idx, start_time, end_time]]  % occupancy N

    Same goes for the gates. Need to chack the idx to verify.
    '''

    FATO_occupancy = [None] * len(node_sequence)
    Gate_occupancy = [None] * len(node_sequence)

    complete_schedule = {}

    for trajectory_name in schedule:

        trajectory = schedule[trajectory_name]
        # Initiate the new trajecotry:
        new_trajectory = [trajectory[0]]

        idx = 0
        for flight in trajectory[1:]:

            idx += 1

            arrival_time    = trajectory[idx-1][2]
            arrival_loc     = trajectory[idx-1][3]
            departure_time  = trajectory[idx][0]
            departure_loc   = trajectory[idx][1]

            N_pax_inbound   = trajectory[idx-1][4]
            N_pax_outbound  = trajectory[idx][4]

            # get the node index
            idx_node    = uF.index_node(node_sequence, departure_loc)
            n_FATOs     = nFnG[idx_node][0]
            n_gates     = nFnG[idx_node][1]
            delta       = nFnG[idx_node][2]

            '''
            Adding the FATO transit legs of the mission:
              x ^
                |                             vehicle i
                |                             /
                |                            /
                |   FATO inb.    FATO out.  /
                |     1____2         5 ____/ 6
                |     /    \         /
                |    /      \_______/ gate 2 fato transit
                |   /       3      4
                |  /        Transit time
                | /
                +--------------------------------------> t

            '''

            # Figure out which fato the vehicle flies to.
                # Question: can we stash a vehicle or is there a conflict?
            t_end_FATO_inbound       = arrival_time + params['FATO_transit_time']


            (FATO_occupancy[idx_node], conflict_flag) = vehicle_placement(FATO_occupancy[idx_node], arrival_time, t_end_FATO_inbound, n_FATOs, params)

            PAV_Id      = flight[5]
            #               pick last value of the data structure

            delta_y1_fato_inb = -(FATO_occupancy[idx_node][-1][0]-1)*delta # we pick the index of the FATO which is occupied and diminish it by one, since the first FATO is in the position that has no decrement.
            delta_y2_fato_inb = -(FATO_occupancy[idx_node][-1][0]-1)*delta # we pick the index of the FATO which is occupied and diminish it by one, since the first FATO is in the position that has no decrement.
            vehicle_states={}
            vehicle_states['delta_1'] = delta_y1_fato_inb
            vehicle_states['delta_2'] = delta_y2_fato_inb

            if conflict_flag == True:
                vehicle_states['conflict'] = True

            FATO_inbound = [arrival_time, arrival_loc, t_end_FATO_inbound, arrival_loc, N_pax_inbound, PAV_Id, vehicle_states]

            # Also need to correct the arrival point of the previous flight (i.e. makes it land using the proper fato instead of going first to top, then going down t right angles.)
            trajectory[idx-1][6]['delta_2'] = delta_y2_fato_inb

            t_start_FATO_outbound       = departure_time - params['FATO_transit_time']

            (FATO_occupancy[idx_node], conflict_flag) = vehicle_placement(FATO_occupancy[idx_node], t_start_FATO_outbound, departure_time, n_FATOs, params)

            delta_y1_fato_out = -(FATO_occupancy[idx_node][-1][0]-1)*delta # we pick the index of the FATO which is occupied and diminish it by one, since the first FATO is in the position that has no decrement.
            delta_y2_fato_out = -(FATO_occupancy[idx_node][-1][0]-1)*delta # we pick the index of the FATO which is occupied and diminish it by one, since the first FATO is in the position that has no decrement.
            vehicle_states={}
            vehicle_states['delta_1'] = delta_y1_fato_out
            vehicle_states['delta_2'] = delta_y2_fato_out

            if conflict_flag == True:
                vehicle_states['conflict'] = True

            FATO_outbound= [t_start_FATO_outbound, departure_loc, departure_time, departure_loc, N_pax_inbound, PAV_Id, vehicle_states]

            # Also need to correct the departure point of the previous flight (i.e. makes it land using the proper fato instead of going first to top, then going down t right angles.)
            trajectory[idx][6]['delta_1'] = delta_y1_fato_out

            # Add a FATO inbound transit leg
            new_trajectory.append(FATO_inbound)

            '''
            Adding the FATO to gate transit legs of the mission:
              x ^
                |                             vehicle i
                |                             /
                |                            /
                |   FATO inb.    FATO out.  /
                |     1____2         5 ____/ 6
                |     /    \         /
                |    /      \_______/ gate 2 fato transit
                |   /       3      4
                |  /        Transit time
                | /
                +--------------------------------------> t

            '''

            # Figure out which gate the vehicle goes to.
            t_start_gate_inbound = t_end_FATO_inbound
            t_end_gate_inbound   = t_start_gate_inbound + params['FATO_to_gate_transit_time']

            t_start_gate_outbound= t_start_FATO_outbound - params['FATO_transit_time']
            t_end_gate_outbound  = t_start_FATO_outbound

            (Gate_occupancy[idx_node], conflict_flag) = vehicle_placement(Gate_occupancy[idx_node], t_end_gate_inbound, t_start_gate_outbound, n_gates, params)

            PAV_Id      = flight[5]
            #               pick last value of the data structure
            vehicle_states={}
            vehicle_states['delta_1'] = delta_y1_fato_inb #starts from the FATO
            vehicle_states['delta_2'] = -Gate_occupancy[idx_node][-1][0]*delta - (n_FATOs-1)*delta # goes to the proper gate

            FATO_to_gate= [t_start_gate_inbound, arrival_loc, t_end_gate_inbound, arrival_loc, N_pax_outbound, PAV_Id, vehicle_states]

            vehicle_states={}
            vehicle_states['delta_1'] = -Gate_occupancy[idx_node][-1][0]*delta - (n_FATOs-1)*delta # goes to the proper gate
            vehicle_states['delta_2'] = delta_y1_fato_out #starts from the FATO

            if conflict_flag == True:
                vehicle_states['conflict'] = True

            gate_to_FATO= [t_start_gate_outbound, arrival_loc, t_end_gate_outbound, arrival_loc, N_pax_outbound, PAV_Id, vehicle_states]

            # Gate waiting time:
            vehicle_states={}
            vehicle_states['delta_1'] = -Gate_occupancy[idx_node][-1][0]*delta - (n_FATOs-1)*delta # goes to the proper gate
            vehicle_states['delta_2'] = -Gate_occupancy[idx_node][-1][0]*delta - (n_FATOs-1)*delta # goes to the proper gate

            Gate_Transit = [t_end_gate_inbound, arrival_loc, t_start_gate_outbound, arrival_loc, N_pax_inbound, PAV_Id, vehicle_states]


            # Add a FATO inbound transit leg
            new_trajectory.append(FATO_to_gate)

            # Add a gate transit leg
            new_trajectory.append(Gate_Transit)

            # Add a FATO outbound transit leg
            new_trajectory.append(gate_to_FATO)

            # Finally, add the outbound FATO segment:
            new_trajectory.append(FATO_outbound)

            new_trajectory.append(flight)

        # Add the completed trajecotry to the completed schedule:
        complete_schedule[trajectory_name] = new_trajectory

    return complete_schedule

def vehicle_placement(FATO_occupancy_i, t_arr, t_dep, n_max_FATO, params):
    '''
    The vehicle placement algorithm places a vehicle on a given FATO or gate
    depending on the occupancy of the vetiport. The sequence that is used
    is always:
    Question 1: are all slots occupied?
        -> If yes, then return the conflict flag (used later to show conflicting
        trajectories) and place the vehicle in the first slot (regardless of conflict)
        -> If no, find the lowest index possible free slot for the vehicle.
    '''

    '''
    Check if the start time is in one of the intervals. This can be replaced with
    a binary tree search, but the current approach is simple and sufficient for
    small schedule:
    '''


    ''' Creation of the data structures containing the occupancies of FATOs and
    gates.

    FATO_occupancy = [  FATO_node_1_occupancy,
                        ...
                        FATO_node_i_occupancy,
                        ...
                        FATO_node_N_occupancy]

    Data structure containing the FATO occupancy for node i:
    FATO_node_i_occupancy =  [[FATO_idx, start_time, end_time],  % occupancy 1
                                  ...                  ,
                              [FATO_idx, start_time, end_time],  % occupancy i
                                  ...                  ,
                              [FATO_idx, start_time, end_time]]  % occupancy N

    Same goes for the gates. Need to chack the idx to verify.
    '''
    overlap_idx_arrival = 0
    overlap_idx_departure=0
    occupied_FATOs = [0] # needs to be initiated at 0, or there is no lower number for in interval in which the find_missing function can looks for missing numbers.

    if not FATO_occupancy_i:
        conflict_flag = 0
        n_fato = 1
        FATO_occupancy_i = [[n_fato, t_arr, t_dep]]

    else:

        for slot in FATO_occupancy_i:
            if t_arr >= slot[1] and t_arr <= slot[2] or t_dep >= slot[1] and t_dep <= slot[2]:
            #if t_arr >= slot[1]+ slot_buffer_time and t_arr <= slot[2]- slot_buffer_time: # if the arrival segment is occupied, the start must be bomped to the next FATO/gate
                overlap_idx_arrival += 1
                occupied_FATOs.append(slot[0])
            if t_dep >= slot[1] and t_dep <= slot[2]: # if the departure segment is occupied, the start must be bomped to the next FATO/gate
            #if t_dep >= slot[1]+ slot_buffer_time and t_dep <= slot[2]- slot_buffer_time: # if the departure segment is occupied, the start must be bomped to the next FATO/gate
                overlap_idx_departure += 1
                occupied_FATOs.append(slot[0])

        occupied_FATOs.append(n_max_FATO+1)
        available_FATO = uF.find_missing(occupied_FATOs)
        if len(available_FATO) == 0:
            conflict_flag = 1
            FATO_occupancy_i.append([1, t_arr, t_dep])

        else:
            conflict_flag = 0
            '''
            Needs to select the FATO that is lowest in index and not in list.
            '''

            lst = occupied_FATOs
            n_fato = uF.find_missing(lst)[0]
            FATO_occupancy_i.append([n_fato, t_arr, t_dep])

    return (FATO_occupancy_i, conflict_flag)

def objective_function(schedule, Network, vehicle_config):

    costs   = 0
    revenue = 0
    list_prices = Network.list_prices
    list_edges  = Network.list_edges

    for trajectory_name in schedule:

        trajectory = schedule[trajectory_name]

        for flight in trajectory:


            start_time= flight[0]
            start_node= flight[1]
            end_time  = flight[2]
            end_node  = flight[3]
            N_pax     = flight[4]
            vehicle_id= flight[5]


            # Cost calculation:
            vehicle = uF.get_vehicle(vehicle_id, vehicle_config)
            vehicle_cost = vehicle['cost']

            flight_time = end_time-start_time

            costs += vehicle_cost*flight_time.total_seconds()/3600 # vehicle cost is in cost per hour

            # Revenue calculation:
            idx_edge = uF.index_edge(list_edges, (start_node, end_node))
            revenue_per_passenger_trip = list_prices[idx_edge]

            revenue += float(N_pax)*revenue_per_passenger_trip


    cash = revenue - costs
    return cash




def evaluate_schedule(json_schedule, vehicle_config, node_list, edge_list, price_list, json_demand, params):

    plt.close("all")

    Network = nW.network(node_list, edge_list, price_list)


    fig, (ax0) = plt.subplots(1)
    Network.plot_network(ax0)

    # Find the earliest and latest dates in the schedule (otherwise plot is done
    # w.r.t to the reference date, which is far too far in the past.)
    date_axis_limits = uF.find_min_max_dates(json_schedule, params)

    '''
    Step 1: Read the schedule out of the json file and load it into the proper
    python data format:
    '''
    schedule = json2schedule(json_schedule, vehicle_config, Network, params)

    '''
    Step 2: Complete the raw schedule with the FATO and gate operations:
    '''
    completeSchedule = complete_schedule(Network, schedule, params)

    '''
    Step 3: Display of the continuity verification function. First in space (vehicle
        departs in wrong place)
    '''
    dis_schedule    = check_continuity(completeSchedule)
    v_schedule      = cH.check_speed(completeSchedule, Network, vehicle_config)
    int_schedule    = cH.check_interference(completeSchedule, Network, vehicle_config) # interference schedule

    n_speed_violations = uF.extract_number_flights(v_schedule)
    n_discontinuities  = uF.extract_number_flights(dis_schedule)
    n_interferences    = uF.extract_number_flights(int_schedule)

    '''
    Step 4: Verify the capacity of the vehicles. Fill in the original schedule
    with a excess capacity value that is plotted. In order to make the constraints
    show more against the rest of the plot, the overcapacity vehicles will be
    plotted using distinct markers. Colors are already used to differentiate
    vehicles.
    '''
    capacitySchedule = cH.check_vehicle_capacity(schedule, Network, vehicle_config)

    '''
    Step 5: Verify the demand profiles for each leg. They must all be positive
    or zero (no passengers invented)
    '''
    panda_Schedule      = pd.DataFrame.from_dict(json_schedule, orient='columns')
    panda_Schedule['sTOT'] =  pd.to_datetime(panda_Schedule['sTOT'], format=params['timeFormat'])
    panda_Schedule['sLDT'] =  pd.to_datetime(panda_Schedule['sLDT'], format=params['timeFormat'])

    panda_demand        = pd.DataFrame.from_dict(json_demand, orient='columns')
    panda_demand['sTOT'] =  pd.to_datetime(panda_demand['sTOT'], format=params['timeFormat'])
    panda_demand['sLDT'] =  pd.to_datetime(panda_demand['sLDT'], format=params['timeFormat'])
    remaining_demand    = check_demand_capacity(panda_Schedule, Network, vehicle_config, panda_demand, params)
    '''
    Step 4: Plot the different graphs:
    The first graph shows the complete schedule as well as the
    '''

    fig, (ax1, ax3) = plt.subplots(2, sharex=True, sharey=True)
    fig.autofmt_xdate()
    plot_schedule(Network, completeSchedule, 'x-t diagram', params, ax1)
    ax1.set_xlim(date_axis_limits)

    fig, (ax1, ax4) = plt.subplots(2, sharex=True, sharey=True)
    fig.autofmt_xdate()
    plot_schedule(Network, completeSchedule, 'x-t diagram', params, ax1)
    ax1.set_xlim(date_axis_limits)


    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    plot_schedule(Network, completeSchedule, 'x-t diagram', params, ax1)
    fig.autofmt_xdate()
    ax1.set_xlim(date_axis_limits)

    fig, (ax1, ax5) = plt.subplots(2, sharex=True, sharey=True)
    plot_schedule(Network, completeSchedule, 'x-t diagram', params, ax1)
    fig.autofmt_xdate()
    ax1.set_xlim(date_axis_limits)

    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
    plot_schedule(Network, completeSchedule, 'x-t diagram', params, ax1)
    fig.autofmt_xdate()
    ax1.set_xlim(date_axis_limits)

    plot_schedule(Network, dis_schedule, 'Discontinuous segments', params, ax2)
    plot_schedule(Network, int_schedule, 'Segments with interference', params, ax3)
    plot_schedule(Network, v_schedule  , 'Speed Violation'       , params, ax4, speed = True)
    plot_schedule(Network, capacitySchedule, 'Excess capacity'       , params, ax5, capacity = True)

    fig, (ax1, ax6) = plt.subplots(2, sharex=True)
    plot_schedule(Network, completeSchedule, 'x-t diagram', params, ax1)
    plot_demand(remaining_demand, params, ax6)
    fig.autofmt_xdate()
    ax1.set_xlim(date_axis_limits)


    profit = objective_function(schedule, Network, vehicle_config)

    # Make a figure of the revenue/cost vs time. Add passengers picked up?

    print('====================================================================')
    print('Schedule evaluation summary:\n')
    print('N. speed violations: {:>30}'.format(n_speed_violations))
    print('N. discontinuities : {:>30}'.format(n_discontinuities))
    print('N. interferences   : {:>30}\n'.format(n_interferences))
    print('Profit             : {:>30}'.format(round(profit,2)))
    print('====================================================================')

    plt.show()

    return 0



