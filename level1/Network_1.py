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



# for node in node_list:
#     if demand['Npax'] >= vehicle['Pax_max']:
#         passenger = vehicle['Pax_max']
#     else:
#         passenger = demand['Npax']



# if rule_A:
#     if rule_A_1:
#         if rule_A_11:
#         elif rule_A_11:
#     elif rule_A_2:
#     else:
# elif rule_B:
#     if rule_B_1:
#     elif rule_B_2:
#         if rule_B_21:
#             print ('too many if!')
#     else:
# elif rule_C:
#     if rule_C_1:
#     else:
# elif rule_D:
#     if rule_D_1:
#     if rule_D_2:
#     else:
# else: