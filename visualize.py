## visualize.py - Parses and plots training log data

import matplotlib.pyplot as plt

with open('training_log.txt') as f:
    iterations = []
    losses = []

    for iteration, data in enumerate(f.readlines()):
        if 'loss:' in data:
            iterations.append(iteration)
            attrs = data.split()
            losses.append(float(attrs[attrs.index('loss:') + 1]))

    plt.plot(iterations, losses, color='red')
    plt.title('Training Performance', fontsize=14)
    plt.xlabel('Iterations', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True)
    plt.show()