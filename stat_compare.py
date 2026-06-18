# This script performs statistical comparisons of model outputs across different configurations.
# For example, df1 and df2 contain aggregated results from two model runs with different parameter settings.

import os
import shutil
import numpy as np
import pandas as pd
import random
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ranksums
cp_dir = '...'#your path of the model output files

from collections import Counter

def most_frequent_item(text_list):
    if not text_list:
        return None  # Return None if the list is empty
    counter = Counter(text_list)
    return counter.most_common(1)[0][0]  # Get the most frequent item

timestamp = '...' # based on the timestamp of your code
TOTAL_SIM_REP_RUNS = 50
total_nr_runs_set = [20,30,40]
code_dominances_set = [0,1,2]#0 = no, 1= low, 2 = high
code_dishonesty_set = [0,1,2]#0 = no, 1= low, 2 = high

results = []
for total_nr_runs in total_nr_runs_set:
    for code_dominances in code_dominances_set:
        for code_dishonesty in code_dishonesty_set:
            for i_REP in range(TOTAL_SIM_REP_RUNS):
                exp_name = 'r'+str(TOTAL_SIM_REP_RUNS)+'_s'+str(total_nr_runs)+'_d'+str(code_dominances)+'_l'+str(code_dishonesty)+'_'
                out_lines_fname = f'exp_{exp_name}out_lines_{timestamp}_{i_REP}.csv'
                #print(out_lines_fname)
                config = f'exp_config_{exp_name}_{timestamp}_{i_REP}.csv'
                #print(config)
                sim = f'sim_{exp_name}_{timestamp}_{i_REP}.csv'
                para_csv = cp_dir+config
                parameters = []
                with open(para_csv, mode='r') as file:
                    reader = csv.reader(file)
                    headers = next(reader)
                    values = next(reader)
                    parameters = dict(zip(headers, values))
                    
                decision_df = pd.read_csv(cp_dir+sim)  # Use header=None if there are no headers
                decision_var = decision_df.iloc[-35:,1]
                np = (decision_var[:20] == 'positive').sum()
                nn = (decision_var[-15:] == 'negative').sum()
                decision_mdl_list = decision_df.iloc[:-35,1]
                decision_mdl = decision_mdl_list.iloc[-1]
                #decision_mdl = decision_mdl_list.value_counts().idxmax()
                results.append({
                    'total_nr_runs': total_nr_runs,
                    'code_dominances': code_dominances,
                    'code_dishonesty': code_dishonesty,
                    'replication_index': i_REP,
                    'nr_runs_mdl':int(parameters['nr_runs_mdl']),
                    'nr_runs_var':int(parameters['nr_runs_var']),
                    'decision_mdl':decision_mdl,
                    'decision_var_p':np,
                    'decision_var_n':nn
                })
df1 = pd.DataFrame(results)
# Display the results DataFrame
print(df1)

import os
import shutil
import numpy as np
import pandas as pd
import random
import csv
from datetime import datetime
import matplotlib.pyplot as plt
# directory = 'C:/Users/ll1d19/OneDrive - University of Southampton/AISD/rag_sim/'
cp_dir = '...'#your path of the model output files

from collections import Counter

def most_frequent_item(text_list):
    if not text_list:
        return None  # Return None if the list is empty
    counter = Counter(text_list)
    return counter.most_common(1)[0][0]  # Get the most frequent item

timestamp = '...' # based on the timestamp of your code
TOTAL_SIM_REP_RUNS = 50
total_nr_runs_set = [20,30,40]
code_dominances_set = [0,1,2]#0 = no, 1= low, 2 = high
code_dishonesty_set = [0,1,2]#0 = no, 1= low, 2 = high

results = []
for total_nr_runs in total_nr_runs_set:
    for code_dominances in code_dominances_set:
        for code_dishonesty in code_dishonesty_set:
            for i_REP in range(TOTAL_SIM_REP_RUNS):
                exp_name = 'r'+str(TOTAL_SIM_REP_RUNS)+'_s'+str(total_nr_runs)+'_d'+str(code_dominances)+'_l'+str(code_dishonesty)+'_'
                out_lines_fname = f'exp_{exp_name}out_lines_{timestamp}_{i_REP}.csv'
                #print(out_lines_fname)
                config = f'exp_config_{exp_name}_{timestamp}_{i_REP}.csv'
                #print(config)
                sim = f'sim_{exp_name}_{timestamp}_{i_REP}.csv'
                para_csv = cp_dir+config
                parameters = []
                with open(para_csv, mode='r') as file:
                    reader = csv.reader(file)
                    headers = next(reader)
                    values = next(reader)
                    parameters = dict(zip(headers, values))
                    
                decision_df = pd.read_csv(cp_dir+sim)  # Use header=None if there are no headers
                decision_var = decision_df.iloc[-35:,1]
                np = (decision_var[:20] == 'positive').sum()
                nn = (decision_var[-15:] == 'negative').sum()
                decision_mdl_list = decision_df.iloc[:-35,1]
                decision_mdl = decision_mdl_list.iloc[-1]
                #decision_mdl = decision_mdl_list.value_counts().idxmax()
                results.append({
                    'total_nr_runs': total_nr_runs,
                    'code_dominances': code_dominances,
                    'code_dishonesty': code_dishonesty,
                    'replication_index': i_REP,
                    'nr_runs_mdl':int(parameters['nr_runs_mdl']),
                    'nr_runs_var':int(parameters['nr_runs_var']),
                    'decision_mdl':decision_mdl,
                    'decision_var_p':np,
                    'decision_var_n':nn
                })
df2 = pd.DataFrame(results)
# Display the results DataFrame
print(df2)


df1["model_good"] = df1["decision_mdl"].isin(["MLP", "SVM"]).astype(int)
df2["model_good"] = df2["decision_mdl"].isin(["MLP", "SVM"]).astype(int)
# Perform Wilcoxon rank-sum test on model choice
statistic, p_value = ranksums(df1["model_good"], df2["model_good"])
print(f"Test Statistic model_good: {statistic}")
print(f"P-Value model_good: {p_value}")
# Perform Wilcoxon rank-sum test on variable selection
statistic, p_value = ranksums(df1["decision_var_p"] - df1["decision_var_n"], df2["decision_var_p"] - df2["decision_var_n"])
print(f"Test Statistic decision_var_p-n: {statistic}")
print(f"P-Value decision_var_p-n: {p_value}")