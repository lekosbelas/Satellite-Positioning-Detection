import matplotlib.pyplot as plt

def plot_loss(loss_values):
    plt.plot(range(1, len(loss_values)+1), loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
