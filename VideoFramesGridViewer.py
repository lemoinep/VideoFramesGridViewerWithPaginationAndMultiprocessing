# Author(s): Dr. Patrick Lemoine

# Video Frames Grid Viewer with Pagination and multiprocessing

# Objective: This program is a video frame viewer of video frames which allows users to travel a video by displaying miniatures of frames arranged in 
# a gate format with pagination. It effectively extracts video frames using multiproaches, showing a selectable number of frames per page.
# Users can navigate between pages via a cursor, mouse wheel or keyboard shortcuts, and click on any sticker to open a detailed view of the selected frame.
# The interface has a personalized appearance, including stylized sliders and labels for an improved user experience. This tool is particularly useful
# for preview and quickly analyze video content by frame.

import sys
import os
import cv2
import numpy as np
import datetime
import configparser
from PyQt5 import QtWidgets, QtGui, QtCore
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def extract_thumbnail(params):
    video_path, frame_no, thumb_size, fps = params
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frame_no, None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return frame_no, None, None
    text_height = 20
    target_height = thumb_size[1] - text_height
    thumb_img = cv2.resize(frame, (thumb_size[0], target_height), interpolation=cv2.INTER_AREA)
    timestamp = str(datetime.timedelta(seconds=frame_no / fps))
    return frame_no, thumb_img, timestamp


class ThumbnailWidget(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(int, np.ndarray)

    def __init__(self, thumb_size, parent=None):
        super().__init__(parent)
        self.frame_number = None
        self.frame_img = None
        self.timestamp = None
        self.thumb_size = thumb_size
        self.setFixedSize(*self.thumb_size)
        self.setStyleSheet("color: white; background-color: #222222; border: 1px solid black;")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("Loading...")

    def set_frame(self, frame_number, frame_img, timestamp):
        self.frame_number = frame_number
        self.frame_img = frame_img
        self.timestamp = timestamp
        self.update_thumbnail()

    def update_thumbnail(self):
        if self.frame_img is None:
            self.setText("Error")
            return
        text_height = 20
        img_height = self.thumb_size[1] - text_height
        img_rgb = cv2.cvtColor(self.frame_img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                            img_rgb.strides[0], QtGui.QImage.Format_RGB888)
        final_img = QtGui.QImage(self.thumb_size[0], self.thumb_size[1], QtGui.QImage.Format_RGB32)
        final_img.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(final_img)
        painter.drawImage(0, 0, qimg)
        painter.setPen(QtGui.QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        text = f"Frame: {self.frame_number}  Time: {self.timestamp}"
        text_rect = QtCore.QRect(0, self.thumb_size[1] - text_height, self.thumb_size[0], text_height)
        painter.drawText(text_rect, QtCore.Qt.AlignCenter, text)
        painter.end()
        pixmap = QtGui.QPixmap.fromImage(final_img)
        self.setPixmap(pixmap)

    def clear_thumbnail(self):
        self.frame_number = None
        self.frame_img = None
        self.timestamp = None
        self.setText("")
        self.setPixmap(QtGui.QPixmap())  # clear image

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.frame_img is not None:
            self.clicked.emit(self.frame_number, self.frame_img)


class FrameViewer(QtWidgets.QWidget):
    def __init__(self, frame_number, frame_img):
        super().__init__()
        self.setWindowTitle(f"Frame {frame_number} Viewer")
        img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        label = QtWidgets.QLabel(self)
        label.setPixmap(pixmap.scaled(800, 600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        self.resize(820, 620)
        self.show()


class MovieGridViewer(QtWidgets.QWidget):
    def __init__(self, video_path, cols, rows, num_workers, max_pages=None):
        super().__init__()
        self.video_path = video_path
        self.num_workers = num_workers

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        #cache_dir_name = video_name.replace(" ", "_").replace(".", "_") + "_cache"
        cache_dir_name = "cache_"+video_name.replace(" ", "_").replace(".", "_") 
        self.cache_dir = os.path.join(video_dir, cache_dir_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.config_path = os.path.join(self.cache_dir, "config.ini")
        self.config = configparser.ConfigParser()
        self.load_or_initialize_config(cols, rows, max_pages)

        self.cols = int(self.config['GRID']['cols'])
        self.rows = int(self.config['GRID']['rows'])
        pages_conf = self.config['GRID']['pages']
        self.max_pages = int(pages_conf) if pages_conf != 'None' else None

        self.setWindowTitle("Video Frames Grid Viewer")
        self.setStyleSheet("background-color: #333333;")

        self.total_per_page = self.cols * self.rows
        self.current_page = 1

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open video file.")
            sys.exit(1)

        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        screen_rect = QtWidgets.QApplication.primaryScreen().availableGeometry()
        screen_width, screen_height = screen_rect.width(), screen_rect.height()

        total_spacing_x = (self.cols - 1) * 10 + 40
        total_spacing_y = (self.rows - 1) * 10 + 70

        thumb_width = (screen_width - total_spacing_x) // self.cols
        thumb_height = int(thumb_width * self.height / self.width) + 20

        total_height = self.rows * thumb_height + total_spacing_y
        if total_height > screen_height:
            scale = screen_height / total_height
            thumb_width = int(thumb_width * scale)
            thumb_height = int(thumb_height * scale)

        self.thumb_size = (thumb_width, thumb_height)

        default_pages = 10
        pages_to_use = self.max_pages if (self.max_pages is not None and self.max_pages > 0) else default_pages
        approx_total_vignettes = self.total_per_page * pages_to_use

        step = max(1, (self.frame_count - 1) // (approx_total_vignettes - 1))
        self.all_frames_to_extract = [i * step for i in range(approx_total_vignettes)]
        self.all_frames_to_extract[-1] = self.frame_count - 1

        calculated_total_pages = (len(self.all_frames_to_extract) + self.total_per_page - 1) // self.total_per_page
        if self.max_pages is not None and self.max_pages > 0:
            self.total_pages = min(calculated_total_pages, self.max_pages)
        else:
            self.total_pages = calculated_total_pages

        max_vignettes = self.total_pages * self.total_per_page
        self.all_frames_to_extract = self.all_frames_to_extract[:max_vignettes]

        self.all_thumbnails_data = []  

        self.v_layout = QtWidgets.QVBoxLayout()
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(20, 20, 20, 20)
        self.v_layout.addLayout(self.grid_layout)

        self.page_thumbnails = []
        for i in range(self.total_per_page):
            thumb = ThumbnailWidget(self.thumb_size)
            row = i // self.cols
            col = i % self.cols
            self.grid_layout.addWidget(thumb, row, col)
            self.page_thumbnails.append(thumb)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setStyleSheet("""
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #b1b1b1, stop:1 #c4c4c4);
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                        stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                        stop:0 #d9d9d9, stop:1 #bfbfbf);
        }
        QSlider::handle:horizontal:pressed {
            background: #ffaa00;
            border: 1px solid #ff8800;
        }
        """)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.v_layout.addWidget(self.slider)

        self.page_label = QtWidgets.QLabel("")
        self.page_label.setAlignment(QtCore.Qt.AlignCenter)
        self.page_label.setStyleSheet("color: white;")
        self.v_layout.addWidget(self.page_label)

        self.setLayout(self.v_layout)

        self.load_thumbnails_from_cache_or_extract()

        self.slider.setMinimum(1)
        self.slider.setMaximum(self.total_pages)
        self.slider.setValue(1)
        self.update_page_label()

        self.showFullScreen()

    def load_or_initialize_config(self, cols, rows, max_pages):
        if os.path.exists(self.config_path):
            self.config.read(self.config_path)
            saved_cols = self.config.getint('GRID', 'cols', fallback=None)
            saved_rows = self.config.getint('GRID', 'rows', fallback=None)
            saved_pages = self.config.get('GRID', 'pages', fallback=None)
            if saved_pages == 'None':
                saved_pages = None
            else:
                saved_pages = int(saved_pages)
            if saved_cols != cols or saved_rows != rows or saved_pages != max_pages:
                self.config['GRID'] = {'cols': str(cols), 'rows': str(rows), 'pages': str(max_pages)}
                with open(self.config_path, 'w') as f:
                    self.config.write(f)
                self.clear_cache()
            else:
                pass
        else:
            self.config['GRID'] = {'cols': str(cols), 'rows': str(rows), 'pages': str(max_pages)}
            with open(self.config_path, 'w') as f:
                self.config.write(f)

    def clear_cache(self):
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".png") and fname != "config.ini":
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                except Exception:
                    pass
        self.all_thumbnails_data = []

    def load_thumbnails_from_cache_or_extract(self):
        loaded_count = 0
        self.all_thumbnails_data = []
        for idx, frame_no in enumerate(self.all_frames_to_extract):
            cache_file = os.path.join(self.cache_dir, f"thumb_{frame_no}.png")
            if os.path.exists(cache_file):

                img = cv2.imread(cache_file)
                timestamp = str(datetime.timedelta(seconds=frame_no / self.fps))
                self.all_thumbnails_data.append((frame_no, img, timestamp))
                loaded_count += 1
            else:
                self.all_thumbnails_data.append((frame_no, None, None))

        if loaded_count < len(self.all_frames_to_extract):
            self.pool = ProcessPoolExecutor(max_workers=self.num_workers)
            self.load_all_thumbnails()
        else:
            self.load_current_page()

    def load_all_thumbnails(self):
        params_list = [(self.video_path, frame_no, self.thumb_size, self.fps) for frame_no in self.all_frames_to_extract]
        self.futures = []
        for i, params in enumerate(params_list):
            future = self.pool.submit(extract_thumbnail, params)
            future.add_done_callback(lambda fut, idx=i: self.handle_loaded(fut, idx))
            self.futures.append(future)

    def handle_loaded(self, fut, idx):
        try:
            frame_no, img, timestamp = fut.result()
        except Exception:
            frame_no, img, timestamp = None, None, None

        if img is not None:
            cache_file = os.path.join(self.cache_dir, f"thumb_{frame_no}.png")
            cv2.imwrite(cache_file, img)

            while len(self.all_thumbnails_data) <= idx:
                self.all_thumbnails_data.append((None, None, None))
            self.all_thumbnails_data[idx] = (frame_no, img, timestamp)

            page_start = (self.current_page - 1) * self.total_per_page
            page_end = page_start + self.total_per_page
            if page_start <= idx < page_end:
                self.update_thumbnail_widget(idx - page_start)

    def update_thumbnail_widget(self, page_idx):
        if 0 <= page_idx < len(self.page_thumbnails):
            global_idx = (self.current_page - 1) * self.total_per_page + page_idx
            if global_idx >= len(self.all_thumbnails_data):
                self.page_thumbnails[page_idx].clear_thumbnail()
                return
            frame_no, img, timestamp = self.all_thumbnails_data[global_idx]
            if img is None:
                self.page_thumbnails[page_idx].clear_thumbnail()
                self.page_thumbnails[page_idx].setText("Loading...")
            else:
                self.page_thumbnails[page_idx].set_frame(frame_no, img, timestamp)
                try:
                    self.page_thumbnails[page_idx].clicked.disconnect()
                except TypeError:
                    pass
                self.page_thumbnails[page_idx].clicked.connect(self.open_frame_viewer)

    def on_slider_change(self, value):
        if value < 1 or value > self.total_pages:
            return
        self.current_page = value
        self.load_current_page()
        self.update_page_label()

    def load_current_page(self):
        page_start = (self.current_page - 1) * self.total_per_page
        for i in range(self.total_per_page):
            global_idx = page_start + i
            if global_idx < len(self.all_thumbnails_data):
                frame_no, img, timestamp = self.all_thumbnails_data[global_idx]
                if img is not None:
                    self.page_thumbnails[i].set_frame(frame_no, img, timestamp)
                    try:
                        self.page_thumbnails[i].clicked.disconnect()
                    except TypeError:
                        pass
                    self.page_thumbnails[i].clicked.connect(self.open_frame_viewer)
                else:
                    self.page_thumbnails[i].setText("Loading...")
                    self.page_thumbnails[i].frame_img = None
            else:
                self.page_thumbnails[i].clear_thumbnail()
                self.page_thumbnails[i].setText("")

    def update_page_label(self):
        self.page_label.setText(f"Page {self.current_page} / {self.total_pages}")

    #def open_frame_viewer(self, frame_number, frame_img):
    #    self.viewer = FrameViewer(frame_number, frame_img)
    #    self.viewer.show()
        
    def open_frame_viewer(self, frame_number, _):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open video to load high-res frame.")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot read frame {frame_number} in high-res.")
            return
        self.viewer = FrameViewer(frame_number, frame)
        self.viewer.show()


    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            if hasattr(self, "pool"):
                self.pool.shutdown(wait=False)
            self.close()
        elif key == QtCore.Qt.Key_Space:
            if self.current_page < self.total_pages:
                self.current_page += 1
                self.slider.setValue(self.current_page)
        elif key == QtCore.Qt.Key_4:
            if self.current_page > 1:
                self.current_page -= 1
                self.slider.setValue(self.current_page)
        elif key == QtCore.Qt.Key_6:
            if self.current_page < self.total_pages:
                self.current_page += 1
                self.slider.setValue(self.current_page)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta < 0:
            if self.current_page < self.total_pages:
                self.current_page += 1
                self.slider.setValue(self.current_page)
                event.accept()
        elif delta > 0:
            if self.current_page > 1:
                self.current_page -= 1
                self.slider.setValue(self.current_page)
                event.accept()
        else:
            super().wheelEvent(event)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video Frames Grid Viewer with Pagination, multiprocessing and cache")
    parser.add_argument("--Path", type=str, required=True, help="Path of video directory")
    parser.add_argument("--Name", type=str, required=True, help="Name of video file")
    parser.add_argument("--Cols", type=int, default=7, help="Number of columns per page")
    parser.add_argument("--Rows", type=int, default=5, help="Number of rows per page")
    parser.add_argument("--Workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--Pages", type=int, default=None, help="Number of pages to display (optional)")
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)  
    app = QtWidgets.QApplication(sys.argv)
    video_full_path = os.path.join(args.Path, args.Name)
    viewer = MovieGridViewer(video_full_path, args.Cols, args.Rows, args.Workers, args.Pages)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
