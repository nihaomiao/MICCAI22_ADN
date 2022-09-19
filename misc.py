import sys
import numpy as np
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout, mode="w"):
        self.terminal = stream
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def map2fig(heatmap, initial=False):
    dpi = 1000.0
    plt.ioff()
    fig = plt.figure(frameon=False)
    if initial:
        heatmap[0, 0] = 1.0
    fig.clf()
    fig.set_size_inches(heatmap.shape[1] / dpi, heatmap.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    cm = plt.cm.get_cmap('jet')
    ax.imshow(heatmap, cmap=cm, aspect='auto')
    fig.set_dpi(int(dpi))
    plt.close(fig)
    return fig2data(fig)[:, :, :3]


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

