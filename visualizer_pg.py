import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

class PGVisualizer(QtWidgets.QWidget):
    update_signal = QtCore.pyqtSignal(object, object)  # particles, estimate

    def __init__(self, beacons, x_range, y_range):
        super().__init__()

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot = self.plot_widget.getPlotItem()
        self.plot.setAspectLocked(True)
        self.plot.setXRange(*x_range)
        self.plot.setYRange(*y_range)
        self.plot.setTitle("Real-Time BLE Particle Filter Localization")

        # Draw beacons
        for bid, (bx, by) in beacons.items():
            self.plot.plot([bx], [by], pen=None, symbol='s', symbolSize=12, symbolBrush='r')
            text = pg.TextItem(text=f"B{bid}", color="r")
            text.setPos(bx + 0.05, by + 0.05)
            self.plot.addItem(text)

        # Create scatter items
        self.particle_scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(0, 0, 255, 120))
        self.estimate_scatter = pg.ScatterPlotItem(size=12, pen=None, brush=pg.mkBrush(0, 255, 0))

        self.plot.addItem(self.particle_scatter)
        self.plot.addItem(self.estimate_scatter)

        # Connect signal to handler
        self.update_signal.connect(self._update_plot)

    def _update_plot(self, particles, estimate):
        """This executes in the Qt GUI thread."""
        if particles is not None:
            self.particle_scatter.setData(particles[:, 0], particles[:, 1])
        if estimate is not None:
            self.estimate_scatter.setData([estimate[0]], [estimate[1]])

    def update(self, particles, estimate):
        """Called from BLE thread."""
        self.update_signal.emit(particles, estimate)