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
from PIL import Image, ImageQt
from pillow_heif import register_heif_opener
import pillow_avif  # AVIF support plugin for Pillow
from PyQt5 import QtWidgets, QtGui, QtCore
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path 

# Register HEIF opener (no register_avif_opener in newer pillow-heif)
register_heif_opener()


def CV_Sharpen2d(source, alpha, gamma, num_op):
    def sharpen_kernel(src):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(src, -1, kernel)

    dst = sharpen_kernel(source)

    if num_op == 1:
        source_filtered = cv2.GaussianBlur(source, (3, 3), 0)
    elif num_op == 2:
        source_filtered = cv2.blur(source, (9, 9))
    else:
        source_filtered = source.copy()

    dst_img = cv2.addWeighted(source_filtered, alpha, dst, 1.0 - alpha, gamma)
    return dst_img

def CV_EnhanceColor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255) 
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)  
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def CV_Vibrance2D(img, saturation_scale=1.3, brightness_scale=1.1, apply=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def view_picture_zoom(img,video_path):
    zoom_scale = 1.0
    zoom_min = 1.0
    zoom_max = 15.0
    mouse_x, mouse_y = -1, -1
    height, width = img.shape[:2]
    qLoop = True
    qSharpen = False
    qEnhanceColor = False
    qVibrance = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal zoom_scale, mouse_x, mouse_y, qLoop 
        mouse_x, mouse_y = x, y
                    
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                zoom_scale = min(zoom_scale + 0.1, zoom_max)
            else:
                zoom_scale = max(zoom_scale - 0.1, zoom_min)
                
        if event == cv2.EVENT_RBUTTONDOWN:
            qLoop = False

                
    def get_zoomed_image(image, scale, center_x, center_y):
        h, w = image.shape[:2]
        new_w = int(w / scale)
        new_h = int(h / scale)

        left = max(center_x - new_w // 2, 0)
        right = min(center_x + new_w // 2, w)
        top = max(center_y - new_h // 2, 0)
        bottom = min(center_y + new_h // 2, h)

        if right - left < new_w:
            if left == 0:
                right = new_w
            elif right == w:
                left = w - new_w
        if bottom - top < new_h:
            if top == 0:
                bottom = new_h
            elif bottom == h:
                top = h - new_h

        cropped = image[top:bottom, left:right]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed
    
    window_name = 'Picture Zoom'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    ratio = width / height
    lh = 800
    lw = int(lh * ratio)
    cv2.resizeWindow(window_name, lw, lh)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    screen_width = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1920  
    screen_height = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) or 1080  

    start_x = int((screen_width - lw) / 2)
    start_y = int((screen_height - lh) / 2)
    cv2.moveWindow(window_name, start_x, start_y)
    
        
    cv2.setMouseCallback(window_name, mouse_callback)

    while qLoop:
        if mouse_x == -1 and mouse_y == -1:
            mouse_x, mouse_y = width // 2, height // 2

        zoomed_img = get_zoomed_image(img, zoom_scale, mouse_x, mouse_y)
        
        if qSharpen:
            zoomed_img = CV_Sharpen2d(zoomed_img, 0.1, 0.0,  1)
            if zoom_scale > 7 :
                zoomed_img = CV_Sharpen2d(zoomed_img, 0.3, 0.0,  1)         
        
        if qEnhanceColor :
            zoomed_img = CV_EnhanceColor(zoomed_img)
        if qVibrance :
            zoomed_img = CV_Vibrance2D(zoomed_img)

        cv2.imshow('Picture Zoom', zoomed_img)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            path = Path(video_path).parent
            new_path = path / "Screenshot"
            new_path .mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            outputName = "Noname"
            outputName = f"{outputName}_{date_time}.jpg"
            outputName = Path(new_path) / outputName
            cv2.imwrite(outputName, zoomed_img)
        elif key == ord('x'):  
            qSharpen = not qSharpen
        elif key == ord('e'):  
            qEnhanceColor = not qEnhanceColor
        elif key == ord('v'):  
            qVibrance = not qVibrance
        elif key == ord('z'):  
            zoom_scale = 1.0
            
    cv2.destroyAllWindows()

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
        view_picture_zoom(img_rgb)
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
    def __init__(self, video_path, cols, rows, num_workers, max_pages=None, thumbnail_format="PNG"):
        super().__init__()
        self.video_path = video_path
        self.num_workers = num_workers
        self.thumbnail_format = thumbnail_format.upper()  # "PNG", "HEIF" or "AVIF"

        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        cache_dir_name = video_name.replace(" ", "_").replace(".", "_") + "_cache"
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
            self.config['GRID'] = {'cols': str(cols), 'rows': str(rows), 'pages': str(max_pages)}
            with open(self.config_path, 'w') as f:
                self.config.write(f)

    def clear_cache(self):
        for fname in os.listdir(self.cache_dir):
            if (fname.endswith(".png") or fname.endswith(".heif") or fname.endswith(".avif")) and fname != "config.ini":
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                except Exception:
                    pass
        self.all_thumbnails_data = []

    def load_thumbnails_from_cache_or_extract(self):
        loaded_count = 0
        self.all_thumbnails_data = []
        ext = self.thumbnail_format.lower()
        for idx, frame_no in enumerate(self.all_frames_to_extract):
            cache_file = os.path.join(self.cache_dir, f"thumb_{frame_no}.{ext}")
            if os.path.exists(cache_file):
                try:
                    pil_img = Image.open(cache_file).convert('RGB')
                    qimg = ImageQt.ImageQt(pil_img)  # Convert PIL Image to QImage
                    # Convert PIL Image to OpenCV BGR ndarray for internal use
                    img = np.array(pil_img)[:, :, ::-1].copy()
                    timestamp = str(datetime.timedelta(seconds=frame_no / self.fps))
                    self.all_thumbnails_data.append((frame_no, img, timestamp))
                    loaded_count += 1
                except Exception as e:
                    print(f"Erreur chargement image {cache_file}: {e}")
                    self.all_thumbnails_data.append((frame_no, None, None))
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
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            ext = self.thumbnail_format.lower()
            cache_file = os.path.join(self.cache_dir, f"thumb_{frame_no}.{ext}")
            try:
                if self.thumbnail_format in ["HEIF", "AVIF"]:
                    pil_img.save(cache_file, format=self.thumbnail_format, quality=90)
                else:
                    pil_img.save(cache_file, format="PNG")
            except Exception as e:
                fallback_path = os.path.join(self.cache_dir, f"thumb_{frame_no}.png")
                pil_img.save(fallback_path, format="PNG")
                ext = "png"
                cache_file = fallback_path
                print(f"Warning: saving thumbnail in {self.thumbnail_format} failed for frame {frame_no}, fallback to PNG. Error: {e}")

            # Reload PIL image for internal use
            try:
                pil_img_reloaded = Image.open(cache_file).convert('RGB')
                img_reloaded = np.array(pil_img_reloaded)[:, :, ::-1].copy()
            except Exception as e:
                print(f"Warning: reloading thumbnail failed for frame {frame_no}: {e}")
                img_reloaded = None

            while len(self.all_thumbnails_data) <= idx:
                self.all_thumbnails_data.append((None, None, None))
            self.all_thumbnails_data[idx] = (frame_no, img_reloaded, timestamp)
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
        
        #self.viewer = FrameViewer(frame_number, frame)
        #self.viewer.show()
        
        view_picture_zoom(frame,self.video_path)


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
    parser.add_argument("--ThumbFormat", type=str, choices=["PNG", "HEIF", "AVIF"], default="PNG",
                        help="Format for saved thumbnails (PNG, HEIF, AVIF)")
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)

    app = QtWidgets.QApplication(sys.argv)
    video_full_path = os.path.join(args.Path, args.Name)
    viewer = MovieGridViewer(video_full_path, args.Cols, args.Rows, args.Workers, args.Pages, args.ThumbFormat)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
