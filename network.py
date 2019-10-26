# -*- coding: utf-8 -*-
'''
This is the file containing all the 'network' classes and functions.
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
import utility_functions as uF

class network:
    ''' Constructor:
    A definition of a network data structure forms the base upon which the x-t
    related functions may query the network for aspect such as length between
    nodes, or for unfolding the network in an efficient way from a 2D
    network to a 1D line.
    A network is defined by a list of nodes and a list of edges.
    The node list has the following structure:

                                A list of tuples
                                        |
                                        V
  node_list = [ (node_name_1, x_coord_1, y_coord_1, n_FATOs_1, n_gates_1 ), ...
                                     ...
                (node_name_i, x_coord_i, y_coord_i, n_FATOs_i, n_gates_i ), ...
                                     ...
                (node_name_N, x_coord_N, y_coord_N, n_FATOs_N, n_gates_N )]

    -node_names are STRINGS
    -x_coords   are DOUBLES
    -y-coords   are DOUBLES
    -n_gates    are INTEGERS
    -n_FATO     are INTEGERS

    This allows the nodes to be displayed in the x-y plane, which is useful for
    debugging purposes amongst other things.
    The edge list has the following structure:

                                A list of tuples
                                        |
                                        V
        edge_list = [ (start_node_name_1, end_node_name_1), ...
                                     ...
                      (start_node_name_j, end_node_name_j), ...
                                     ...
                      (start_node_name_M, end_node_name_M)]

      -node_names are STRINGS (and have to be part of the node list! If they are
                               not, an error is thrown until they are)

      Note 1: there can be many ways to expand this definition of network, such
      as distances between nodes which might not be euclidian. But for now, the
      strcuture is left as simple as possible.

      Note 2: Here, we assume that start_nodes and end_nodes are
      interchangeable, i.e. they edges do not have a direction.

      Note 3: edges can be defined multiple times. This needs to be prevented,
      thus an enumerating function takes care of this when the network
      is being defined.

      Note 4: Nods might be completely disconnected from the network. If that
      is the case, they have no use. This is to be flagged to the user when he
      defines the network. ***14.9.2019: To be coded.***
    '''
    def __init__(self, list_nodes, list_edges, list_prices):
        self.list_nodes = list_nodes
        self.list_edges = list_edges
        self.list_prices= list_prices

        #----------------------------------------------------------------------
        # Check of the names in the list of edges:
        # First the names of the nodes are extracted from the edges ...
        start_node_name = [item[0] for item in self.list_edges]
        end_node_name   = [item[1] for item in self.list_edges]
        node_names      = [item[0] for item in self.list_nodes]

        # ... then they are compared to the names in the node list:
        diff_start  = np.setdiff1d(start_node_name, node_names)
        diff_end    = np.setdiff1d(end_node_name, node_names)

        if len(diff_start) > 0:
            raise ValueError('One of the start nodes of an edge does not exist in network')

        if len(diff_end) > 0:
            raise ValueError('One of the end nodes of an edge does not exist in network')

        #----------------------------------------------------------------------
        # Check for redundant edges within the network:
        # First step is to sort all the tuples (so that (A,B) = (B,A))...
        sorted_edge_list = [tuple(sorted(item)) for item in self.list_edges]
        # ... then a conversion to a dictionary is used to search all other
        # possible edges that might be the same in O(n)
        # see:
        # https://www.geeksforgeeks.org/python-program-to-count-duplicates-in-a-list-of-tuples/
        count_map = {}
        for i in sorted_edge_list:
            count_map[i] = count_map.get(i, 0) +1
            if count_map.get(i, 0) > 1:
                raise ValueError('There is a redundant edge in the network')

        # Need to add here: the check of non-connected nodes. To be added? O/C 14.9.2019

    def get_nodes(self):
        return self.list_nodes

    def get_edges(self):
        return self.list_edges

    def get_distance(self, node_0, node_1):
        '''
        GET_DISTANCE returns the distance between 2 given nodes. It also has the
        function of flagging whether the distance is existant or not (i.e.
        whether the edge linking the two nodes is part of the network definition
        or not.) If it is not the case, then the connectivity flag is raised
        (= 1). However, the distance is still calculated.

        For the current version of this code, the distance is calculated as being
        the Euclidian distance between the points. In further versions the
        definition of the edges may be expanded in order to include the flight
        distance between the nodes as a new parameter:
        '''

        idx_0 = uF.index_node(self.list_nodes, node_0)
        idx_1 = uF.index_node(self.list_nodes, node_1)

        edge_d0 = (node_0, node_1) # name of the edge in one direction...
        edge_d1 = (node_1, node_0) # ...and name of the edge in the other direction

        if edge_d0 in self.list_edges or edge_d1 in self.list_edges: # of the edge exists in the network:
            connect = 1
        else: # if the edge does not exist in the network:
            connect = 0

        # Calculate the postions. Note that we divide by 1000,  in order to avoid an overflow warning:
        norm_val = 1000.0
        pos_0 = np.array([self.list_nodes[idx_0][1], self.list_nodes[idx_0][2]])/norm_val
        pos_1 = np.array([self.list_nodes[idx_1][1], self.list_nodes[idx_1][2]])/norm_val

        d = norm_val*sum((pos_1 - pos_0)**2)**0.5

        return (d, connect)

    '''plot_network:

        The function works in the following steps:
            1) it sets up the figure alone (set_title, set_xlabel, etc.)
            2) it loops through the list of nodes and plots them as a small
               disc with their name next to it
            3) it loops through the edges in order to plot them.
    '''
    def plot_network(self, ax=None,**kwargs):
        # creates a figure with specified number so that existiing figures are not overwritten
        ax = ax or plt.gca()

        ax.set_title('Network Top View')

        ax.set_xlabel('x distance')
        ax.set_ylabel('y distance')

        for node in self.list_nodes:
            #  add  x-coord, y-coord, name
            ax.text(node[1], node[2], node[0], fontsize=15)
            ax.plot(node[1], node[2], 'o', color='b')

        for edge in self.list_edges:
            # finds index, then extracts the coordinates, then plots them:
            index_start_node    = uF.index_node(self.list_nodes, edge[0])
            index_end_node      = uF.index_node(self.list_nodes, edge[1])
            start_node          = self.list_nodes[index_start_node]
            end_node            = self.list_nodes[index_end_node]
            # plots a straight line betwee the 2 nodes:
            #           x_start       x_end           y_start       y_end
            ax.plot([start_node[1], end_node[1]], [start_node[2], end_node[2]], '--', color='k')

#            # Plot the distance between nodes on the graph:
#            ax.text(node[1], node[2], node[0], fontsize=15)
        return ax

def NetworkTo1D(Network):
    '''
    The NetworkTo1D function takes the 2 dimensional network as an input as
    turns it to a sequence of nodes (turns it to a 1D sequence) that can
    be put on the y-axis of an x-t plot.
    INPUTS:
        Network: network is the class written above. It describes the
                 transportation network to be modeled.

    OUTPUTS:
        Sequence: this is the sequence of vertiports (by name) that allow to
                  describe all edges of the network on a single axis. For
                  example, the network below:

                      A ---- B-----D
                      \     /
                       \   /
                        \ /
                         C
                Would be described by the sequence [[D-B-C-A-B]] so that all 4
                edges are represented as contiguously as possible. If there
                are more than a single Euler path in the graph, there are
                more than a single list in the list (see example below)

    Function:
        From an algorithmic point of view, the problem described here is a
        Eulerian Path Problem
        https://en.wikipedia.org/wiki/Eulerian_path
        The algorithm used here is the Fleury algorithm, although it performs
        only in O(n**2). It has the merit of being simple to implement (easy
        to debug) and easy to tailor to the current pupose.
        The Euler theorem states that a graph (or network) has a Eulerian Cycle
        (i.e. a path exists where all edges may be visited only once and one
        returns to ones departure point) if all vertices have an even number
        of edges connected to it.
        A graph may have at most 2 vertices with an odd number of edges if
        a Eulerian Path exists (i.e. a path exists where all edges may be
        visited only once but the departure and arrival point are different).

        Although there are many ways to improve upon the method used subsequently,
        this simple approach is used because it allows for a fast initial
        implementation. Since this function stands alone, it can be perfected
        without impacting the rest of the code.

        The sequence used is:
            1) Verify that the network has a Euler path. If not, flag it.
            2) If there is a Euler circle (all vertices are even), choose any
            starting point (in this case the first one in the node list of the
            network)
            3) If ther is a Euler path (Only 2 vertices are odd) then pick
            one of the 2 vertices as the starting point. The other will be
            automatically the end-point
            4) If there is not Euler path, run algorithm as in 3), and then
            append the last edges on top of the others like in this example:

                A-------B
                |\     /|
                | \   / |
                |  \ /  |
                |   X   |
                |  / \  |
                | /   \ |
                |/     \|
                D-------C

            -Case 4) applies.
            -Pick A as starting point.
            -Go through network using a Fleury algorithm

            A---B
               /|
              / |
             /  |
            D---C

            -Once end achieved start from remaining graph as if it were new
            (Remaining network is shown below:)

            A   B
            |\ 
            | \
            |  \
            D   C

            Final sequence is:

            [A, B, C, D, B]+[D, A, C]= [[A,B,C,D,B],[D,A,C]]

    '''
    # First step: classify network into 1) Euler cycle, 2) Euler path, 3) Euler
    # paths.
    # Data handling: convert the network format to an array where occurences
    # may be counted easily:
    list_edges = Network.get_edges()
    start_node_list = [item[0] for item in list_edges]
    end_node_list   = [item[1] for item in list_edges]

    # By concatenating both lists, all the tips of the edges are contained into
    # a single list, which can be queried:
    node_list       = start_node_list + end_node_list

    # The Counter function produces a dictionnary that contains the frequency
    # each element appears.This is equivalent to the order of the corresponding
    # vertex:
    freq = collections.Counter(node_list)

    (even_list, odd_list) = Split(freq.items())

    # Now the dictionnary can be quieried to see if all the vertices have
    # even numbers or not. If all vertices are even, then the graph has a
    # Euler Cycle. If only 2 of them are odd, then a complete Euler path is
    # possible:

    if bool(odd_list) == False:
        #print 'The network has a Euler cycle'
        '''
        If the graph has a Eulerian cycle, any point on the graph may be
        picked as the starting point. For ease of coding, we shall use the
        first node in the list as the start/finish point.
        '''
        edge_list       = Network.get_edges()[:] # the [:] is essential to pass the list by value and not reference
        idx_start_edge  = 0
        Sequence = Fleury(edge_list, idx_start_edge)

    elif len(odd_list) < 3:
        print('The network has a Euler path')
    else:
        print('The network has neither single Euler path nor cycle')

        ''' NOTE:
            If there is no Euler path or Euler cycle, then the graph is not
            displayed. Handling non-Eulerian graphs is to be implemented as
            soon as the Eulerian graphs are handeled properly.
            '''

    return Sequence

def Split(list_vertices):
    ''' SPLIT is used by the NETWORKTO1D function in order to evaluate whether
    a network has a Euler path, Euler cycle or none.
    It works by taking as input the dictionnary of vertices from a network with
    their number of edges, ad splitting them into 2 lists, one with an even
    number of vertices, and one with an odd number of vertices.
    INPUTS:
        list_vertices: this is a python list of tuples, of the form:

    list_vetices = [ (node_name_1, number_of_adjacent_edges_1 ), ...
                             ...
                     (node_name_2, number_of_adjacent_edges_2 ), ...
                             ...
                     (node_name_N, number_of_adjacent_edges_N )]

    OUTPUTS:
        ev_li:  a python list of tuples of vertices that have a even number of
                edges
        od_li:  a python list of tuples of vertices that have odd number of
                edges
    '''
    ev_li = []
    od_li = []
    for i in list_vertices:
        # element i is composed of i = ('VertexName', edge_number). So
        # i[1]%2 == 0 evaluates if the number of edges is odd or even:
        if (i[1] % 2 == 0):
            ev_li.append(i)
        else:
            od_li.append(i)

    return (ev_li, od_li)

def Fleury(edge_list, idx_start_edge):
    '''
    The Fleury Algorithm is designed to seep walk across a network in order to
    organise the edges into a line. The resulting path can be represented as
    a sequence, which can be plotted on the y-axis of an x-t diagram.

       A                   y ^
     /  \          =>        |
    /    \                 A +
   B------C                  |
                           B +
                             |
                           C +
                             |
                           A +-----------------------------> t

    The algorithm below "walks" across the edge list in an ordonated fashion.
    It starts at a starting edge (input to the function) using the index
    of the edge (idx_start_edge).
    Then it moves across the edge list of the network:
        1) after starting at a node, it "crosses" the edge by changing its
                current node from the start node of the edge to the arrival
                of the edge. Note that edges do not have a direction, so the
                arrival node may be in first or second position in the data
                structure:
                    (A, B) is equivalent to (B, A)
                So to solve this problem, the departing node is removed from the
                tuple (which has to be converted to a list prior):
                    (A, B), start node is A -> (A,B) -> [A,B]-> [B]
        2) The arrival node is the new start node. But at this point, we have
                not chosen an edge that departs from the arrival node yet.
                So the edge_list is iterated through until an edge is found
                which contains the current node.

    INPUTS:
        edge_list: The edge list has the following structure:

                                A list of tuples
                                        |
                                        V
        edge_list = [ (start_node_name_1, end_node_name_1), ...
                                     ...
                      (start_node_name_j, end_node_name_j), ...
                                     ...
                      (start_node_name_M, end_node_name_M)]


        idx_start_edge: the edge from which the walk starts. Implicitly, this
        is actually an edge AND a node, since by default the start_node of the
        edge is used as the departing node.

    OUTPUTS:
        node_sequence: is the node sequence is a list of the vertices, defining
                        the Euler path or cycle.

    '''
    # Initiate node_sequence as an empty list:
    node_sequence = []

    # The start edge is specified in the input:
    idx_edge = idx_start_edge

    # We assume the the start node is the first defined in the edge.
    # selects the start node from which the Fleury algorithm starts:
    start_node = edge_list[idx_start_edge][0]

    # The start node is appended to the node sequence:
    node_sequence.append(start_node)

    # While there are still edges in the list, i.e edges in the network, the
    # algorithm runs:
    while bool(edge_list) == True:
        # Selects the current edge:
        current_edge = list(edge_list[idx_edge])

        # Once the node has been crossed, it is removed from the remaining edge,
        # leaving the arrival node as only element:
        current_edge.remove(start_node)

        # Next node is the node remaining in the edge node pair:
        next_node = current_edge[0]

        # Once the edge has been crossed, it is removed from the network:
        edge_list.remove(edge_list[idx_edge])
        # The edge index starts as negative in order to have the interator
        # incremented at the beginning of the loop, before the break happens:
        idx_edge = -1

        # Loop skins edges in search of the next edge to start from the current
        # node:
        for edge in edge_list:
            idx_edge += 1
            # once an edge is found with the correct node, the loop is stopped
            # and the index of the edge is passed upward to start at the while
            # loop again.
            if next_node in edge:
                node_sequence.append(next_node)
                start_node = next_node

                # add the final node in memory:
                final_edge = list(edge)
                final_edge.remove(next_node)
                final_node = final_edge[0]
                break
            # if the index is the length of the remaining list of edges, it
            # means that node has no departing edges from it. The sequence must
            # thus be stopped:
            elif idx_edge + 1 == len(edge_list):
                edge_list = []

                break

    node_sequence.append(final_node)

    return node_sequence
