"""
For plotting graph output

"""

import matplotlib.pyplot as plt


def plot_results(losses, val_losses, accs, val_accs, epochs=100):
    # Plot training results
    fig = plt.figure(figsize=(15, 5))
    axs = fig.add_subplot(1, 2, 1)
    axs.set_title('Loss')
    # Plot all metrics
    axs.plot(range(epochs), losses, label='loss')
    axs.plot(range(epochs), val_losses, label='val loss')
    axs.legend()

    axs = fig.add_subplot(1, 2, 2)
    axs.set_title('Accuracy')
    # Plot all metrics
    axs.plot(range(epochs), accs, label='acc')
    axs.plot(range(epochs), val_accs, label='val acc')
    axs.legend()

    plt.show()
