# -*- coding: utf-8 -*-
"""
The parameter file is supposed to contain some general function parameters that
are specific to the execution and plotting of the results, such as plot parameters
(spacing etc.)

Note: the definition of the names needs to be consisitent.

"""
from datetime import timedelta

# Plotting parameters:
params = {}

params['h_max']     = 0.5 # maximum height that a vertiport may occupy in the visualization.
params['delta']     = 0.1 # default vertical distance between fatos and gates during the representation.
params['linewidth']     = 2
params['figFontSize']   = 11
params['linestyle']     = '-'
params['timeFormat']    = '%Y/%m/%dT%H:%M:%SZ' #'%Y-%m-%dT%H:%M:%SZ' # previous time format
params['plot_axis_time_buffer'] = timedelta(minutes = 5) # buffer time added before and after the earliest and latest date for the timeline plotting.
# Trajecotry parameters:

#params['FATO_transit_time'] = 45 # FATO transit time estimated at 45 seconds.
#params['FATO_to_gate_transit_time'] = 45

params['FATO_transit_time'] = timedelta(seconds = 0)
params['FATO_to_gate_transit_time'] = timedelta(seconds = 0)
params['FATO_slot_buffer_time'] = timedelta(seconds = 120)
params['time_epsilon'] = timedelta(seconds = 1) # micro time difference to differentiate segments that have the same time.
