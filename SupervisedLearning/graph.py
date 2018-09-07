import matplotlib.pyplot as plt


def plot(x_axis, y_axis, title=None):
    plt.plot(x_axis, y_axis, 'g^')
    if title:
        plt.title(title)
    plt.show()
