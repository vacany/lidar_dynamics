import vispy
from vispy import visuals, app
from vispy.scene import SceneCanvas
import numpy as np
from matplotlib import pyplot as plt


class Visual_PCL_Template:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self):
        pass

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor=(0.5, 0.5, 0.5))
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = vispy.scene.visuals.Markers()

        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)
        vispy.scene.visuals.XYZAxis(parent=self.scan_view.scene)

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()

        vispy.app.quit()

    def run(self):
        vispy.app.run()



class Vis_function(Visual_PCL_Template):
    def __init__(self, dataset, function, config, offset=0):
        super().__init__()
        self.dataset = dataset
        self.function = function

        self.original_config = config.copy()
        self.config = config

        self.offset = offset
        self.total = len(dataset)

        # value init
        self.keys = list(self.config.keys())
        self.cur_key = 0
        self.cur_value = self.config[self.keys[self.cur_key]]

        self.reset()
        self.update_scan()
        self.run()

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def update_scan(self):

        data = self.dataset.get_multi_frame(self.offset)

        self.config[self.keys[self.cur_key]] = self.cur_value

        range_data = self.function(data, self.config)
        range_data = range_data / range_data.max()

        # then change names
        title = f"{self.keys[self.cur_key]} \t ---> {self.cur_value:.2f} \t Scan {self.offset} out of {self.total}"
        self.canvas.title = title


        print(f"\r \n{self.keys[self.cur_key]}\n \t {self.cur_value:.2f}", end='')

        print_str = [f"{self.keys[i]} \t {self.config[self.keys[i]]}\n" for i in range(len(self.keys))]
        text_to_print = "".join(print_str)
        # print(f'\r{text_to_print}', end='')
        # print(self.config)

        viridis_range = np.array((range_data * 255), np.uint8)
        viridis_map = self.get_mpl_colormap("jet")
        viridis_colors = viridis_map[viridis_range]

        data['points'][:, :3] -= data['points'][:, :3].mean(0)

        self.scan_vis.set_data(data['points'][:, :3],
                               face_color=viridis_colors[..., ::-1],
                               edge_color=viridis_colors[..., ::-1],
                               size=3)


    def key_press(self, event):
        self.canvas.events.key_press.block()
        # self.img_canvas.events.key_press.block()

        # if event.key == 'S' and self.metric is not None:
        #     self.metric.print_stats()

        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0

        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1

        # Shift by 50 frames
        elif event.key == '0':
            self.offset += 50

        elif event.key == '9':
            self.offset -= 50

        elif event.key == '-':
            self.cur_value -= 0.05

        elif event.key == '=':
            self.cur_value += 0.05

        elif event.key == 'k':
            self.cur_key += 1
            if self.cur_key >= len(self.keys):
                self.cur_key = 0

            self.cur_value = self.config[self.keys[self.cur_key]]

        elif event.key == 'r':
            self.config = self.original_config

        self.update_scan()

        if event.key == 'Q' or event.key == 'Escape':
            self.destroy()
