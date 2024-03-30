import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

original_data = {
    "English": [541.6],
    "Finnish": [32.3],
    "Code":    [207.5], 
    "Eng-Fin": [8.0], 
    }

sampled_data = {
    "English": [541.6],
    "Finnish": [129.1],
    "Code":    [315.4], 
    "Eng-Fin": [8.0], 
    }

# Create a function to display the percentage and the original value
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{v:d}B ({p:.1f}%)'.format(p=pct,v=val)
    return my_format

for data, name in ((original_data, 'original'), (sampled_data, 'sampled')):
    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(12, 12))

    # Variables to keep track of the cumulative size
    total_size = sum([value[0] for value in data.values()])
    labels = list(data.keys())

    #colors = sns.color_palette("Set2")
    colors = [
        (0xbb/256, 0xde/256, 0x94/256, 1.0),
        (0xae/256, 0xcd/256, 0xe0/256, 1.0),
        (0xed/256, 0x9f/256, 0x9b/256, 1.0),
        (0xf3/256, 0xc2/256, 0x7c/256, 1.0),
    ]

    # Create the donut plot
    wedges, texts, autotexts = ax.pie(
        [value[0] for value in data.values()],
        labels=labels,
        colors=colors,
        autopct=autopct_format([value[0] for value in data.values()]),
        startangle=140,
        pctdistance=0.75,
        labeldistance=None,
        wedgeprops=dict(width=0.5),
        textprops={'fontsize': 34}
    )

    # Add a legend
    # ax.legend(wedges, data.keys(),
    #           loc="center right",
    #           bbox_to_anchor=(0, 0.5),
    #           fontsize=15)  # Increase the fontsize to make the legend bigger

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"data_distribution_{name}.pdf")
