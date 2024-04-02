import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data
fertility = {
    "Poro 34B": [1.38, 1.06, 1.00, 1.15],
    "Llama 33B": [3.14, 1.26, 1.35, 1.92],
    "MPT 30B": [2.85, 1.08, 1.18, 1.70],
    "Falcon 40B": [2.95, 1.09, 1.26, 1.77],
    "FinGPT": [1.23, 1.49, 1.89, 1.53],
    "StarCoder": [3.19, 1.29, 1.15, 1.88],
}

order = ["Poro 34B", "Llama 33B", "MPT 30B", "Falcon 40B", "FinGPT", "StarCoder"]

groups = ["Finnish", "English", "Code", "Average"]

# Create a new figure with a specific size (width, height)
plt.figure(figsize=(12, 3))

plt.rc('font', size=10)    # defaults

# colors = [
#     (0x49/256, 0x9f/256, 0xf8/256),
#     (0x81/256, 0xd5/256, 0x53/256),
#     (0x54/256, 0xaf/256, 0x32/256),
#     (0x24/256, 0x8f/256, 0x16/256),
#     #(0x35/256, 0x79/256, 0x76/256),
#     (0x33/256, 0x74/256, 0xb5/256),
#     #(0xef/256, 0xbd/256, 0x40/256),
#     (0xd4/256, 0x70/256, 0xa4/256),
# ]

colors = [
    #(0x49/256, 0x9f/256, 0xf8/256),
    (0x59/256, 0xaf/256, 0xff/256),
    (0x91/256, 0xe5/256, 0x63/256),
    (0x64/256, 0xbf/256, 0x42/256),
    (0x24/256, 0x8f/256, 0x16/256),
    (0x38/256, 0x84/256, 0xc5/256),
    (0xd4/256, 0x70/256, 0xa4/256),
]

# Set the width of the bars and the positions of the bars on the x-axis
barWidth = 0.75
groupWidth = len(fertility) * barWidth
spacing = 0.7  # Add a spacing variable

# ???
rs = []
rs.append(np.arange(len(fertility["Poro 34B"])) * (groupWidth + spacing))
for i in range(len(order)-1):
    rs.append([x + barWidth for x in rs[-1]])

# Create the bars
bars = []
for i, n in enumerate(order):
    bars.append(plt.bar(
        rs[i], 
        [min(value, 20) for value in fertility[n]],
        width=barWidth,
        color=colors[i],
        edgecolor='white',
        label=n
    ))

# Function to add value labels
def add_labels(bars, original_values):
    for bar, original_value in zip(bars, original_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(original_value), ha='center', va='bottom', fontsize=11)

# Add labels to the bars
for i, n in enumerate(order):
    add_labels(bars[i], fertility[n])

# Add labels, title, and legend
plt.xticks([r + groupWidth/2 - barWidth/2 for r in rs[0]], groups, fontsize=12)  # Decrease the size of the x-ticks labels
plt.legend(bbox_to_anchor=(1.1, 1.05))

# Remove the plot borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Remove the y-axis ticks
plt.yticks([])

# Remove the y-axis ticks and labels
plt.tick_params(axis='y', which='both', length=0, labelleft=False)

# Show the plot
plt.tight_layout()
plt.savefig("fertility_plot.pdf")
