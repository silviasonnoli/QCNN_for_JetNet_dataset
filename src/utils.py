import os
import matplotlib.pyplot as plt
import numpy as np

def print_training_history(history, **kwargs):
    plots_dir = '.plots'
    os.makedirs(plots_dir, exist_ok=True)
    color_list = ['blue', 'red', 'green', 'grey']

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    all_lines = []

    for i, (metric_name, y) in enumerate(history.items()):
    
        if metric_name.find("accuracy") > -1:
            ax = ax1
        else:
            ax = ax2
        line, = ax.plot(np.linspace(1, len(y), len(y)), y, 
                    color=color_list[i], 
                    label=metric_name, 
                    linestyle='-')
    
    all_lines.append(line)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax1.legend(all_handles, all_labels, loc='upper left')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='black')
    ax2.set_ylabel('Loss', color='black')

    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='black')

    labels = [l.get_label() for l in all_lines]

    plt.title(f"Accuracy and loss for {kwargs['model_type']}")
    fig.tight_layout()
    fig_name = "training"
    for v in kwargs.values():
        fig_name += f"_{v}"
    fig_name += ".png"
    fig.savefig(os.path.join(plots_dir, fig_name))
    plt.show()
