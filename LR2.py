"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""

import math
import os
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
from sklearn.cluster import KMeans


def rotate3d(pos, roll, pitch, yaw):
    """
    Rotates point on angle
    :param pos: (X, Y, Z) tuple position
    :param roll: Roll angle in radians
    :param pitch: Pitch angle in radians
    :param yaw: Yaw angle in radians
    :return:
    """
    roll_sin = math.sin(roll)
    roll_cos = math.cos(roll)
    pitch_sin = math.sin(pitch)
    pitch_cos = math.cos(pitch)
    yaw_sin = math.sin(yaw)
    yaw_cos = math.cos(yaw)

    mat_roll = np.array([[1, 0, 0], [0, roll_cos, -roll_sin], [0, roll_sin, roll_cos]])
    mat_pitch = np.array([[pitch_cos, 0, pitch_sin], [0, 1, 0], [-pitch_sin, 0, pitch_cos]])
    mat_yaw = np.array([[yaw_cos, -yaw_sin, 0], [yaw_sin, yaw_cos, 0], [0, 0, 1]])

    pos = np.array(pos)

    pos = np.dot(pos, mat_roll)
    pos = np.dot(pos, mat_pitch)
    pos = np.dot(pos, mat_yaw)

    return pos


def inside_test(points, cube3d):
    """
    Checks the location of points relative to the cube
    Code from https://stackoverflow.com/a/53559963
    :param points: array of points with shape (N, 3)
    :param cube3d: numpy array of the shape (8,3) with coordinates in the clockwise order.
    first the bottom plane is considered then the top one
    :return: indices of the points array which are outside the cube3d
    """
    b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

    dir1 = (t1 - b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2 - b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b4 - b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t3) / 2.0

    dir_vec = points - cube3d_center

    res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) > size1)[0]
    res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) > size2)[0]
    res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) > size3)[0]

    return list(set().union(res1, res2, res3))


def list_direction(data):
    """
    Determines whether an array is increasing or decreasing
    :param data:
    :return:
    """
    inc_points = 0
    dec_points = 0
    for i in range(len(data)):
        average = sum(data[i:]) / len(data[i:])
        if data[i] < average:
            inc_points += 1
        elif data[i] > average:
            dec_points += 1

    if inc_points > dec_points:
        return 1
    elif dec_points > inc_points:
        return 0
    else:
        return None


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        # Load GUI file
        uic.loadUi('LR2.ui', self)

        # Laser angles
        self.laser_angles = [-30.67, -9.33, -29.33, -8.00, -28.00, -6.66, -26.66, -5.33, -25.33, -4.00, -24.00, -2.67,
                             -22.67, -1.33, -21.33, 0.00, -20.00, 1.33, -18.67, 2.67, -17.33, 4.00, -16.00, 5.33,
                             -14.67, 6.67, -13.33, 8.00, -12.00, 9.33, -10.67, 10.67]

        # System variables
        self.dump_file = None
        self.reader_running = False
        self.dump_paused = False
        self.points = []
        self.distances = []
        self.points_surface = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]))
        self.cube_lines = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 0]]))
        self.corridor_lines = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 0]]))
        self.estimator = KMeans(n_clusters=3)

        # Connect GUI controls
        self.btn_load_data.clicked.connect(self.load_data)
        self.btn_stop_reading.clicked.connect(self.stop_reading)
        self.btn_pause.clicked.connect(self.pause)
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_opengl)
        self.plot_timer.start(100)

        # Initialize table
        self.init_tables()

        # Initialize openGL view
        self.init_opengl()

        # Show GUI
        self.show()

    def init_tables(self):
        """
        Initializes table of packets and setup table (whitelist table)
        :return:
        """
        self.points_table.setColumnCount(3)
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.points_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Packet'))
        self.points_table.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Yaw'))
        self.points_table.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Distances'))
        header = self.points_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

    def init_opengl(self):
        """
        Initializes charts
        :return:
        """
        self.openGLWidget.addItem(gl.GLAxisItem())
        self.openGLWidget.addItem(self.points_surface)
        self.openGLWidget.addItem(self.cube_lines)
        self.openGLWidget.addItem(self.corridor_lines)

    def update_opengl(self):
        """
        Draws points over openGL view
        :return:
        """
        if len(self.points) > 0:

            # Height color map
            color_data = np.array([item[2] for item in self.points])

            if self.radio_cmap_dist.isChecked():
                # Default color map
                color_data = np.array(self.distances)
            elif self.radio_cmap_clusters.isChecked():
                if not self.dump_paused:
                    # Fit into k-means estimator if enabled
                    self.estimator = KMeans(n_clusters=int(self.spin_clusters.value()))
                    self.estimator.fit(np.array(self.points))
                if hasattr(self.estimator, 'labels_'):
                    # Clusters color map
                    color_data = np.array(self.estimator.labels_)

            color_map = plt.get_cmap('hsv')
            min_data = np.min(color_data)
            max_data = np.max(color_data)
            if max_data == min_data:
                max_data = 1
            rgba_img = color_map((color_data - min_data) / (max_data - min_data))

            self.points_surface.setData(pos=np.array(self.points), color=rgba_img,
                                        size=2, pxMode=True)

            cube_w = self.slider_cube_w.value() / 2
            cube_h = self.slider_cube_h.value() / 2
            cube_d = self.slider_cube_d.value() / 2

            tl_f = [-cube_w, cube_h, -cube_d]
            tr_f = [cube_w, cube_h, -cube_d]
            br_f = [cube_w, -cube_h, -cube_d]
            bl_f = [-cube_w, -cube_h, -cube_d]
            tl_n = [-cube_w, cube_h, cube_d]
            tr_n = [cube_w, cube_h, cube_d]
            br_n = [cube_w, -cube_h, cube_d]
            bl_n = [-cube_w, -cube_h, cube_d]

            cube_points = [tl_f, tr_f, br_f, bl_f, bl_n, tl_n, tl_f, bl_f, bl_n, br_n, tr_n, tr_f, br_f, br_n, tr_n,
                           tl_n]

            self.cube_lines.setData(pos=np.array(cube_points), color=[1, 1, 1, 1])

            corr_yaw = self.slider_corr_yaw.value()
            corr_x = - self.slider_corr_x.value() / 100
            corr_w = self.slider_corr_w.value() / 10 + 0.1
            corr_h = self.slider_corr_h.value() / 50 + 0.1
            corr_d = self.slider_corr_d.value() / 50 + 0.1

            tl_f = rotate3d([-corr_w, corr_h, -corr_d], 0, 0, math.radians(corr_yaw))
            tr_f = rotate3d([corr_x, corr_h, -corr_d], 0, 0, math.radians(corr_yaw))
            br_f = rotate3d([corr_x, -corr_h, -corr_d], 0, 0, math.radians(corr_yaw))
            bl_f = rotate3d([-corr_w, -corr_h, -corr_d], 0, 0, math.radians(corr_yaw))
            tl_n = rotate3d([-corr_w, corr_h, corr_d], 0, 0, math.radians(corr_yaw))
            tr_n = rotate3d([corr_x, corr_h, corr_d], 0, 0, math.radians(corr_yaw))
            br_n = rotate3d([corr_x, -corr_h, corr_d], 0, 0, math.radians(corr_yaw))
            bl_n = rotate3d([-corr_w, -corr_h, corr_d], 0, 0, math.radians(corr_yaw))

            # Check obstacles
            cube_test = (bl_f, br_f, br_n, bl_n, tl_f, tr_f, tr_n, tl_n)
            outside_ids = inside_test(np.array(self.points), cube_test)
            density = (len(self.points) - len(outside_ids)) / len(self.points)
            density *= 100
            if density > 1:
                density = 1

            corr_points = [tl_f, tr_f, br_f, bl_f, bl_n, tl_n, tl_f, bl_f, bl_n, br_n, tr_n, tr_f, br_f, br_n, tr_n,
                           tl_n]

            self.corridor_lines.setData(pos=np.array(corr_points), color=[density, 1 - density, 0, 1])

    def load_data(self):
        """
        Loads dump file
        :return:
        """
        if not self.reader_running:
            if os.path.exists(self.data_file.text()):
                print('Loading data...')
                self.dump_file = open(self.data_file.text(), 'rb')
                self.reader_running = True
                thread = threading.Thread(target=self.dump_reader)
                thread.start()
            else:
                print('File', self.data_file.text(), 'doesn\'t exist!')

    def pause(self):
        """
        Pauses data stream
        :return:
        """
        self.dump_paused = not self.dump_paused
        if self.dump_paused:
            self.btn_pause.setText('Resume')
        else:
            self.btn_pause.setText('Pause')

    def stop_reading(self):
        """
        Stops reading data from dump file
        :return:
        """
        self.reader_running = False
        self.dump_file.close()

    def dump_reader(self):
        """
        Reads dump from file
        :return:
        """
        # Clear table and data arrays
        self.points_table.setRowCount(0)

        # Create temp buffers
        bytes_buffer = [0] * 100
        bytes_buffer_position = 0
        previous_byte = 0
        packets_read = 0
        self.distances = [0.] * 360 * 32
        self.points = [[0., 0., 0.]] * 360 * 32
        # Array of indexes
        indexes = np.array([[0] * 32] * 360)
        index = 0
        for yaw in range(360):
            for pitch in range(32):
                indexes[yaw][pitch] = index
                index += 1

        # Continue reading
        while self.reader_running:
            incoming_bytes = self.dump_file.read(1024)
            if incoming_bytes is None or len(incoming_bytes) == 0:
                self.reader_running = False
                break

            for incoming_byte in incoming_bytes:

                while self.dump_paused:
                    time.sleep(0.1)

                bytes_buffer[bytes_buffer_position] = incoming_byte
                if bytes_buffer[bytes_buffer_position] == 238 and previous_byte == 255:
                    bytes_buffer_position = 0
                    if bytes_buffer.count(0) < len(bytes_buffer) - 2:
                        # Check for not 00 packet

                        # Calculate yaw angle
                        laser_yaw = (int(bytes_buffer[1] & 0xFF) << 8) | int(bytes_buffer[0] & 0xFF)
                        laser_yaw /= 100.0
                        while laser_yaw >= 360:
                            laser_yaw -= 360.0

                        first_distance = 0
                        last_distance = 0
                        byte_position = 2
                        laser_num = 0
                        while byte_position <= 96:
                            # Calculate distance
                            distance = (int(bytes_buffer[byte_position + 1] & 0xFF) << 8) | int(
                                bytes_buffer[byte_position] & 0xFF)
                            distance *= 0.002

                            # First and last distances for table
                            if laser_num == 0:
                                first_distance = distance
                            elif laser_num == 31:
                                last_distance = distance

                            if distance < 50:
                                # Calculate pitch angle
                                laser_pitch = self.laser_angles[laser_num]

                                # Rotate distance over pith and yaw
                                rotated_point = rotate3d((distance, 0, 0), 0, math.radians(laser_pitch),
                                                         math.radians(laser_yaw))

                                # Find flat array index
                                array_position = indexes[int(laser_yaw)][laser_num]

                                # Check cube and FOV
                                cube_w = self.slider_cube_w.value() / 2
                                cube_h = self.slider_cube_h.value() / 2
                                cube_d = self.slider_cube_d.value() / 2
                                if not (-cube_w < rotated_point[0] < cube_w and
                                        -cube_h < rotated_point[1] < cube_h and
                                        -cube_d < rotated_point[2] < cube_d and
                                        abs(laser_pitch) <= self.spin_fov_pitch.value() and
                                        laser_yaw <= self.spin_fov_yaw.value()):
                                    rotated_point = [0, 0, 0]
                                    distance = 0

                                # Fill arrays
                                self.points[array_position] = \
                                    [rotated_point[0], rotated_point[1], rotated_point[2]]
                                self.distances[array_position] = distance

                            byte_position += 3
                            laser_num += 1

                        # Write packet to the table
                        row_number = self.points_table.rowCount()
                        self.points_table.insertRow(row_number)
                        self.points_table.setItem(row_number, 0, QTableWidgetItem(str(row_number)))
                        self.points_table.setItem(row_number, 1, QTableWidgetItem(str(laser_yaw)))
                        self.points_table.setItem(row_number, 2, QTableWidgetItem(str(first_distance) +
                                                                                  ' ... ' + str(last_distance)))
                        packets_read += 1
                        # time.sleep(0.001)
                else:
                    previous_byte = bytes_buffer[bytes_buffer_position]
                    bytes_buffer_position += 1
                    if bytes_buffer_position >= 100:
                        bytes_buffer_position = 0

        self.dump_file.close()
        print('File reading stopped. Read', packets_read, 'packets')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    win = Window()
    sys.exit(app.exec_())
