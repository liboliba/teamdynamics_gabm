# This script provides data visualisation of the model results.
import os
import shutil
import numpy as np
import pandas as pd
import random
import csv
from datetime import datetime
import matplotlib.pyplot as plt
cp_dir = '...' # your file path

from collections import Counter

def most_frequent_item(text_list):
    if not text_list:
        return None  # Return None if the list is empty
    counter = Counter(text_list)
    return counter.most_common(1)[0][0]  # Get the most frequent item

timestamp = '...' #your timestamp 
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
results_df = pd.DataFrame(results)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from collections import defaultdict

# Labels
dominance_labels = {0: 'No', 1: 'Low', 2: 'High'}
dishonesty_labels = {0: 'No', 1: 'Low', 2: 'High'}
run_labels = [20, 30, 40]

# Model colors
all_models = sorted(results_df['decision_mdl'].dropna().unique())
colors = plt.cm.Set2(np.linspace(0, 1, len(all_models)))
color_map = dict(zip(all_models, colors))

# Compute average proportion across all subplots for sorting legend
prop_sum = defaultdict(float)
prop_count = defaultdict(int)

for run in run_labels:
    for d in range(3):
        for h in range(3):
            df = results_df[
                (results_df['total_nr_runs'] == run) &
                (results_df['code_dominances'] == d) &
                (results_df['code_dishonesty'] == h)
            ]
            if not df.empty:
                freq = df['decision_mdl'].value_counts(normalize=True)
                for mdl, val in freq.items():
                    prop_sum[mdl] += val
                    prop_count[mdl] += 1

mean_prop = {mdl: prop_sum[mdl]/prop_count[mdl] for mdl in all_models}
# Sort models by results
sorted_models = sorted(all_models, key=lambda m: mean_prop[m], reverse=True)

# Adjustable line spacing
line_spacing = 0.25
y_top = 0.7
label_offset = 0.02

fig, axs = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey=True)
axs = axs.flatten()

for i, (dominance, dishonesty) in enumerate(
    [(d, h) for d in range(3) for h in range(3)]
):
    ax = axs[i]

    for j, run in enumerate(run_labels):
        df = results_df[
            (results_df['total_nr_runs'] == run) &
            (results_df['code_dominances'] == dominance) &
            (results_df['code_dishonesty'] == dishonesty)
        ]

        if not df.empty:
            freq = df['decision_mdl'].value_counts(normalize=True)
            freq = freq.sort_values(ascending=False)

            left = 0
            y_pos = y_top - j*line_spacing

            for mdl, prop in freq.items():
                ax.hlines(y=y_pos, xmin=left, xmax=left+prop, colors=color_map[mdl], linewidth=4)
                left += prop

            if dishonesty == 0:
                ax.text(-0.02, y_pos + label_offset, str(run), va='center', ha='right',
                        fontsize=9, transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")

# labels
for row in range(3):
    axs[row*3].set_ylabel(f'Dominance: {dominance_labels[row]}', rotation=0,
                          labelpad=60, fontsize=10, va='center')

for col in range(3):
    axs[col].set_title(f'Dishonesty: {dishonesty_labels[col]}', fontsize=10, pad=10)

# Shared legend sorted by mean proportion
handles = [Line2D([0], [0], color=color_map[m], linewidth=4) for m in sorted_models]
fig.legend(handles, sorted_models, loc='center right', title='Model', frameon=False)

plt.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.savefig("cdf_all_2026.png", dpi=300, bbox_inches="tight")
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_df['percentage_p'] = (results_df['decision_var_p'] / 20) * 100
results_df['percentage_n'] = (results_df['decision_var_n'] / 15) * 100
results_df['percentage_total'] = ((results_df['decision_var_p'] + results_df['decision_var_n']) / 35) * 100

label_map = {0: 'No', 1: 'Low', 2: 'High'}
results_df['Dominance'] = results_df['code_dominances'].map(label_map)
results_df['Dishonesty'] = results_df['code_dishonesty'].map(label_map)

df_plot = results_df.melt(
    id_vars=['total_nr_runs', 'Dominance', 'Dishonesty'],
    value_vars=['percentage_p', 'percentage_n', 'percentage_total'],
    var_name='Metric', value_name='Accuracy'
)
df_plot['Metric'] = df_plot['Metric'].replace({
    'percentage_p': 'Positive', 'percentage_n': 'Negative', 'percentage_total': 'Total'
})

sns.set_theme(style="whitegrid")
order = ['No', 'Low', 'High']

g = sns.catplot(
    data=df_plot, x='Metric', y='Accuracy', hue='total_nr_runs',
    row='Dominance', col='Dishonesty', row_order=order, col_order=order,
    kind='box', 
    width=0.7,            # Slightly wider boxes look better in thin plots
    dodge=True,           
    palette='viridis',
    height=3.5,           # Tall height
    aspect=0.8,           # THIN: width is only 80% of height
    margin_titles=False, 
    fliersize=2
)

g.set_titles("")
g.set_xlabels("")
g.set_ylabels("")

for ax, label in zip(g.axes[0, :], order):
    ax.set_title(f"Dishonesty: {label}", fontweight='normal', size=11, pad=12)

for i, row_axes in enumerate(g.axes):
    ax_left = row_axes[0]
    ax_left.set_ylabel(f"Dominance: {order[i]}", fontweight='normal', size=11, labelpad=40)
    ax_left.yaxis.set_tick_params(labelleft=True, labelsize=10)

g.figure.text(0.14, 0.5, 'Accuracy (%)', va='center', ha='center', 
              rotation='vertical', fontweight='normal', size=11)

sns.move_legend(
    g, "lower center", 
    bbox_to_anchor=(0.5, 0.94), 
    ncol=3, 
    title="Run Length", 
    frameon=True
)

g.set(ylim=(0, 105))
g.despine(left=False)
plt.subplots_adjust(top=0.90, bottom=0.08, left=0.20, right=0.98, hspace=0.3)
plt.savefig("variable_selection.png", dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd

results_df['positive'] = (results_df['decision_var_p'] / 20) * 100
results_df['negative'] = (results_df['decision_var_n'] / 15) * 100
results_df['total'] = ((results_df['decision_var_p'] + results_df['decision_var_n']) / 35) * 100

label_map = {0: 'No', 1: 'Low', 2: 'High'}
order = ['No', 'Low', 'High']

results_df['Dominance'] = pd.Categorical(results_df['code_dominances'].map(label_map), categories=order)
results_df['Dishonesty'] = pd.Categorical(results_df['code_dishonesty'].map(label_map), categories=order)

metrics = ['positive', 'negative', 'total']
summary = results_df.groupby(['total_nr_runs', 'Dominance', 'Dishonesty'], observed=True)[metrics].agg(['mean', 'std'])

report_df = pd.DataFrame()

for col in metrics:
    # Creating the "Mean +- Std" string
    report_df[col] = summary[col].apply(
        lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1
    )

report_df = report_df.reset_index()
report_df.columns = ['Runs', 'Dominance', 'Dishonesty', 'Positive (%)', 'Negative (%)', 'Total (%)']
print(report_df.to_string(index=False))
report_df.to_csv("summary_table_accuracy.csv", index=False)