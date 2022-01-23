## visualize.py - Parses and plots training log data

import matplotlib.pyplot as plt
import pandas as pd
import textwrap as twp
from collections import defaultdict

with open('training_log.txt') as f:
    ''' Plot training log '''
    iterations = []
    losses = []

    for iteration, data in enumerate(f.readlines()):
        if 'loss:' in data:
            iterations.append(iteration)
            attrs = data.split()
            losses.append(float(attrs[attrs.index('loss:') + 1]))

    plot1 = plt.figure(1)
    plt.plot(iterations, losses, color='red')
    plt.title('Training Performance', fontsize=14)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True)

with open('eval.txt') as f:
    ''' Make table of evaluation metrics '''
    fig, ax = plt.subplots()
    metrics = set()
    metric_stats = defaultdict(list)

    for data in f.readlines():
        attrs = data.split()
        metric = ' '.join(attrs[:3])
        metrics.add(metric)
        metric_stats[metric].append(twp.fill(' '.join(attrs[3:]), 30))

    for metric in metric_stats:
        metric_stats[metric] = metric_stats[metric][-3:]
    
    df = pd.DataFrame(metric_stats, columns=list(metrics))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()

plt.show()