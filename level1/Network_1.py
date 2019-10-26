''' Polyhack NETWORK 1: this network has a Euler cycle:

     A-----B
     |    /
     |   /
     |  /
     | /
     |/
     C

# First the node list is declared, see network structure in the fleetPath
  script:'''
node_list = [('A', 0,     20000, 1, 1),
             ('C', 26500, 34000, 2, 3),
             ('B', 63000, 0,     1, 2)]

edge_list = [('A', 'B'),
             ('B', 'C'),
             ('C', 'A')]

price_list= [33, 45, 55] # price per passenger per leg
