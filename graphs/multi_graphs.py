
from glob import glob
import matplotlib.pyplot as plt
import os


from graphs.func_graph import create_graph

# directory path that contains all the folders of pkl files
directory_path = os.path.expanduser('~') + '/graphs'


#### performance graphs ####
result_folders = glob(directory_path + "/*/")

# initilizes the config of plt
#colors = ['tab:orange','#21618C','r','g', 'tab:orange', 'c', 'y','m','#21618C']
#colors = ['b','m', 'r', 'g','tab:orange', 'c', 'y','m','#21618C']
#colors = ['r','#21618C', 'g', 'b','tab:orange', 'c', 'y','m','#21618C']
colors = ['tab:orange','#21618C','r', 'm', 'c', 'y']
#colors = ['darkcyan','tab:red', 'darkblue', '#21618C', 'm', 'c', 'y']
#colors = ['k','b','m', 'r', 'g', 'c', 'y','m','#21618C']
#colors = ['k','g','r','m', 'c', 'y','m','#21618C']
#colors = ['k','r','r','m', 'c', 'y','m','#21618C']
smoothness=2
f = plt.figure(1)
create_graph(plt=plt, target="mod_reward", plt_figure= f, y_label="Reward", x_label="Number of Steps", result_folders=result_folders
             , colors=colors,smoothness=smoothness, directory_path = directory_path, name = 'Performance')
plt.show()
#### exploration graph in polyrl ####
g = plt.figure(2)
create_graph(plt=plt, target="poly_exploration", plt_figure= g, y_label="target policy percentage", x_label="Number of Steps", result_folders=result_folders
             , colors=colors,smoothness=0, folder_name_cons="poly", directory_path = directory_path, name = 'Percentage')


