''' Polyhack NETWORK 2: this network has no single Euler path (minimum 2):


     F
     |       G----------H
     |      /
     A-----B
     |    /\
     |   /  \
     |  /    \
     | /      \
     |/        \
     E----------C
      \         |
       \        |
        \       |
         \      |
          D-----I

# First the node list is declared, see network structure in the fleetPath
  script:'''
node_list = [('A', 0,      20000, 1, 2),
             ('B', 2650,   34000, 3, 5),
             ('C', 63000,      0, 2, 3),
             ('D', 45000, -21000, 2, 2),
             ('E', 32000,  10000, 3, 5),
             ('F',-10000,  25000, 3, 5),
             ('G', 3000,   42000, 3, 5),
             ('H', 50000,  40000, 3, 5),
             ('I', 64000, -22000, 2, 3)]

edge_list = [('A', 'B'),
             ('B', 'C'),
             ('A', 'E'),
             ('E', 'B'),
             ('E', 'C'),
             ('E', 'D'),
             ('A', 'F'),
             ('B', 'G'),
             ('G', 'H'),
             ('C', 'I'),
             ('D', 'I')]

price_list= [33,
             50,
             33,
             20,
             33,
             33,
             33, # FA same as AB
             15, # BG about half that of AB
             50, # GH about same as BC
             33, # CI about same as AE
             33] # DI about same as AE
              # price per passenger per leg


















