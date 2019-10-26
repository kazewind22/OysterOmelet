''' Polyhack NETWORK 2: this network has no single Euler path (minimum 2):
    
     A-----B
     |    /\ 
     |   /  \
     |  /    \
     | /      \
     |/        \
     E----------C
      \
       \
        \
         \
          D
          
# First the node list is declared, see network structure in the fleetPath 
  script:'''
node_list = [('A', 0,      20000, 1, 2),
             ('B', 2650,   34000, 1, 2),
             ('C', 63000,      0, 2, 2),
             ('D', 45000, -21000, 2, 2),
             ('E', 32000,  10000, 3, 5)]

edge_list = [('A', 'B'),
             ('B', 'C'),
             ('A', 'E'),
             ('E', 'B'),
             ('E', 'C'),
             ('E', 'D')]

price_list= [33, 
             50, 
             33,
             20,
             33,
             33] # price per passenger per leg
