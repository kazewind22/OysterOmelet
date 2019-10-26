# -*- coding: utf-8 -*-
import fleet_path
import json
import matplotlib.pyplot as plt
import pandas as pd
import network as nW

# Uncomment the level that you are currently evaluating:
#==============================================================================
# LEVEL 1:
json_schedule_filename          = './level1/example_schedule.json'
json_vehicle_config_filename    = './level1/vehicle.json'
parameter_filename              = './level1/parameters.py'
network_filename                = './level1/Network_1.py'
demand_filename                 = './level1/demand.json'
#==============================================================================

##==============================================================================
## LEVEL 2:
##json_schedule_filename          = './level2/yourSchedule_2.json'
#json_vehicle_config_filename    = './level2/vehicle.json'
#parameter_filename              = './level2/parameters.py'
#network_filename                = './level2/Network_2.py'
#demand_filename                 = './level2/demand_2.json'
##==============================================================================

##==============================================================================
## LEVEL 3:
##json_schedule_filename          = './level3/yourSchedule_3.json'
#json_vehicle_config_filename    = './level3/vehicle.json'
#parameter_filename              = './level3/parameters.py'
#network_filename                = './level3/Network_3.py'
#demand_filename                 = './level3/demand_3.json'
##==============================================================================


with open(json_schedule_filename) as data_file:
    Schedule = json.load(data_file)

with open(json_vehicle_config_filename) as data_file:
    vehicle_config = json.load(data_file)

with open(demand_filename) as data_file:
    demand = json.load(data_file)

exec(open(parameter_filename).read()) # contains params
exec(open(network_filename).read())   # contains node_list, edge_list, price_list

fleet_path.evaluate_schedule(Schedule, vehicle_config, node_list, edge_list, price_list, demand, params)
