# Run heatmap_creator_polished.py to generate the heatmap
# import heatmap_creator_polished
# heatmap_creator_polished.main()

import heatmap_creator_polished_ss3
heatmap_creator_polished_ss3.main()

# Run results_csv_analysis.py to generate csv with all the results
import results_csv_analysis
results_csv_analysis.main()

import tl_overview_plot
tl_overview_plot.main()

import tl_performance_yeild_plot
tl_performance_yeild_plot.main()

import layer_analysis_plot
layer_analysis_plot.main()