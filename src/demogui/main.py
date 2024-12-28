import kp
import cv2, os, shutil, sys
from enum import Enum
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QComboBox, QFileDialog, QMessageBox, QHBoxLayout, QDialog, QListWidget,
                             QScrollArea, QFrame, QListWidgetItem, QTextEdit)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo, QMediaRecorder, QAudioRecorder
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QTimer, QUrl
from demogui.utils import perform_inference,preprocess_image #functions from utils.py

# Constants
UXUI_ASSETS = "../../uxui/"
WINDOW_SIZE = (1200, 900)
BACKGROUND_COLOR = "#143058"
SECONDARY_COLOR = "#1260E6"
DEVICE_BOX_STYLE = f"background-color: {BACKGROUND_COLOR}; padding: 20px; border-radius: 20px; padding: 10px 20px;"
BUTTON_STYLE = """
    QPushButton {
        background: transparent;
        color: white;
        border: 1px solid white;
        border-radius: 20px;
        padding: 10px 20px;
    }
    QPushButton:hover {
        background-color: rgba(255, 255, 255, 50);
    }
    QPushButton:pressed {
        background-color: rgba(255, 255, 255, 100);
    }
"""
SQUARE_BUTTON_STYLE = "background: transparent; color: white; border: 1px transparent; border-radius: 10px; "
POPUP_SIZE_RATIO = 0.67
NO_DEVICE_GIF = UXUI_ASSETS + "no_device_temp.gif"


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model_buttons = [
            ('Face Detection', self.run_face_detection),
            ('Gender/Age Detection', self.run_gender_age_detection),
            ('Object Detection', self.run_object_detection),
            ('Mask Detection', self.run_mask_detection),
            ('Image Project', self.start_image_project),
            ('Upload Model', self.upload_model)
        ]

        self.connected_devices = [
        ]

        self.input_directory = ""
        self.to_keep_directory = ""
        self.to_delete_directory = ""
        self.image_directory = ""
        self.label_directory = ""

        self.video_widget = QVideoWidget(self)
        self.camera = QCamera(QCameraInfo.defaultCamera())
        self.image_capture = QCameraImageCapture(self.camera)
        self.media_recorder = QMediaRecorder(self.camera)
        self.audio_recorder = QAudioRecorder(self)  
        self.camera.setViewfinder(self.video_widget)

        self.right_layout = QVBoxLayout()
        self.left_layout = QVBoxLayout()
    

# TODO: find the correct mapping of the values
    class K_(Enum):
        KL520 = 256
        KL720 = 720
        KL720_L = 512 #legacy
        KL530 = 530
        KL832 = 832
        KL730 = 732
        KL630 = 630
        KL540 = 540


    def init_ui(self):
        self.setGeometry(100, 100, *WINDOW_SIZE)
        self.setWindowTitle('Innovedus AI Playground')
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR};")
        self.layout = QVBoxLayout(self)

        self.show_welcome_label()
        QTimer.singleShot(5000, self.show_device_popup_and_main_page)


    def show_welcome_label(self):
        welcome_label = QLabel(self)
        welcome_pixmap = QPixmap(UXUI_ASSETS + "kneron_logo.png")
        welcome_label.setPixmap(welcome_pixmap)
        welcome_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(welcome_label)


    def close_connection_page(self):
        print("closing device connection page")
        device_descriptors = kp.core.scan_devices()
        if device_descriptors.device_descriptor_number > 0:
            for device in device_descriptors.device_descriptor_list:
                self.parse_and_store_devices(device_descriptors.device_descriptor_list)
                kp.core.connect_devices(usb_port_ids=[device.usb_port_id])
        self.load_firmware()
        self.popup_window.close()

    
    def load_firmware(self):
        print("loading firmware")
        for device in self.connected_devices:
            device_group = kp.core.connect_devices(usb_port_ids=[device.get["usb_port_id"]])
            kp.core.set_timeout(device_group=device_group, milliseconds=5000)
            SCPU_FW_PATH = f'../../external/res/firmware/{device.get["product_id"]}/fw_scpu.bin'
            NCPU_FW_PATH = f'../../external/res/firmware/{device.get["product_id"]}/fw_ncpu.bin'
            kp.core.load_firmware_from_file(device_group=device_group,
                                            scpu_fw_path=SCPU_FW_PATH,
                                            ncpu_fw_path=NCPU_FW_PATH)
    

# TODO: load models on multiple dongles (needs new kp api version)
    def load_models(self, device_group, model_path):
        print("loading models")
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                        file_path=model_path)

# TODO: implement function. general post processing of raw results
    def run_inference(self, image_file_path, device_group, model_nef_descriptor):
        img_bgr565 = preprocess_image(image_file_path)
        inf_node_output_list = perform_inference(device_group, model_nef_descriptor, img_bgr565)
        return inf_node_output_list


    def show_error_popup(self, message):
        error_dialog = QMessageBox.critical(self, "Error", message)
        

    def parse_and_store_devices(self, devices):
        for device in devices:
            new_device = {
                'usb_port_id': device.usb_port_id,
                'product_id': device.product_id,
                'kn_number': device.kn_number
            }
            print(device)
            existing_device_index = next((index for (index, d) in enumerate(self.connected_devices) 
                                        if d['usb_port_id'] == new_device['usb_port_id']), None)
            
            if existing_device_index is not None:
                self.connected_devices[existing_device_index] = new_device
            else:
                self.connected_devices.append(new_device)


    def check_available_device(self):
        print("checking available devices")
        device_descriptors = kp.core.scan_devices()
        self.clear_device_layout(self.device_layout)

        if device_descriptors.device_descriptor_number > 0:
            if device_descriptors.device_descriptor_number > 0:
                self.parse_and_store_devices(device_descriptors.device_descriptor_list)
                self.display_devices(device_descriptors.device_descriptor_list)
        else:
            self.show_no_device_gif()


    def get_dongle_type(self, product_id):
        for dongle_type in self.K_:
            if dongle_type.value == product_id:
                return dongle_type
        return None

        
    def display_devices(self, device_descriptor_list):
        hbox_layout = QHBoxLayout()
        hbox_layout.setAlignment(Qt.AlignCenter)

        for device in device_descriptor_list:
            device_layout = QVBoxLayout()
            box_layout = QVBoxLayout()

            icon = QLabel()
            pixmap = QPixmap(UXUI_ASSETS + "kneron_logo.png")
            icon.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            print(device.product_id)
            usb_type_label = QLabel(f"Device: {self.get_dongle_type(device.product_id)}")
            usb_type_label.setAlignment(Qt.AlignCenter)
            usb_type_label.setStyleSheet("color: white")

            box_layout.addWidget(icon, alignment=Qt.AlignCenter)
            box_layout.addWidget(usb_type_label)

            box_widget = QWidget()
            box_widget.setLayout(box_layout)

            box_size = 200 
            box_widget.setFixedSize(box_size, box_size)
            box_widget.setStyleSheet(DEVICE_BOX_STYLE)

            usb_port_label = QLabel(f"KN number:\n{device.kn_number}")
            usb_port_label.setAlignment(Qt.AlignLeft)
            usb_port_label.setStyleSheet("color: white;") 

            label_icon_layout = QHBoxLayout()

            small_icon = QSvgWidget(UXUI_ASSETS + "./Assets_svg/btn_dialog_device_disconnect_normal.svg")
            small_icon.setFixedSize(30, 30)

            label_icon_layout.addWidget(usb_port_label)
            label_icon_layout.addWidget(small_icon, alignment=Qt.AlignRight)

            device_layout.addWidget(box_widget)
            device_layout.addLayout(label_icon_layout)

            device_widget = QWidget()
            device_widget.setLayout(device_layout)

            hbox_layout.addWidget(device_widget)

        self.device_layout.addLayout(hbox_layout)



    def show_no_device_gif(self):
        no_device_label = QLabel(self)
        no_device_movie = QMovie(NO_DEVICE_GIF)
        no_device_label.setMovie(no_device_movie)
        no_device_movie.start()
        no_device_label.setAlignment(Qt.AlignCenter)
        self.device_layout.addWidget(no_device_label)


    def show_device_connection_popup(self):
        self.popup_window = QDialog(self)
        self.popup_window.setWindowTitle("Device Connection")
        self.popup_window.setFocusPolicy(Qt.StrongFocus)

        popup_width = int(self.width() * POPUP_SIZE_RATIO)
        popup_height = int(self.height() * POPUP_SIZE_RATIO)
        self.popup_window.setGeometry(100, 100, popup_width, popup_height)
        self.popup_window.setStyleSheet(f"background-color: {SECONDARY_COLOR};")

        popup_layout = QVBoxLayout()

        self.device_layout = QVBoxLayout()  

        popup_title = QHBoxLayout()
        small_icon = QSvgWidget(UXUI_ASSETS + "./Assets_svg/ic_window_device.svg")
        small_icon.setFixedSize(30, 30)
        popup_title.addWidget(small_icon)
        device_popup_label = QLabel("Device Connection", self.popup_window)
        device_popup_label.setAlignment(Qt.AlignCenter)
        popup_title.addWidget(device_popup_label)
        popup_layout.addLayout(self.device_layout)

        button_layout = QHBoxLayout()

        refresh_button = QPushButton('Refresh')
        refresh_button.clicked.connect(self.check_available_device)
        refresh_button.setStyleSheet(BUTTON_STYLE)
        button_layout.addWidget(refresh_button)

        done_button = QPushButton('Done')
        done_button.setStyleSheet(BUTTON_STYLE)
        #done_button.clicked.connect(self.close_connection_page)
        done_button.clicked.connect(lambda: self.close_connection_page())
        
        button_layout.addWidget(done_button) 

        popup_layout.addLayout(button_layout)
        self.popup_window.setLayout(popup_layout)

        self.popup_window.setModal(True)
        self.setEnabled(False)  
        self.popup_window.finished.connect(lambda: self.setEnabled(True))

        self.popup_window.show()

        self.check_available_device()


    def show_device_popup_and_main_page(self):
        self.show_device_connection_popup()
        self.popup_window.finished.connect(self.main_page)



    def clear_device_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()

    
    def clear_layout(self):
        for i in reversed(range(self.device_layout.count())): 
            self.device_layout.itemAt(i).widget().deleteLater()


    def create_frame(self, title, icon_path):
        frame = QFrame(self)
        frame.setStyleSheet(f"border: none; background: {SECONDARY_COLOR}; border-radius: 20px;")
        layout = QVBoxLayout(frame)

        title_layout = QHBoxLayout()
        title_icon = QSvgWidget(icon_path)
        title_icon.setFixedSize(40, 40)
        title_layout.addWidget(title_icon)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: white;")
        title_layout.addWidget(title_label)

        layout.addLayout(title_layout)
        return frame
    

    def add_model_buttons(self, layout):
        for model_name, run_function in self.model_buttons:
            button = QPushButton(model_name)
            button.clicked.connect(run_function)
            
            button.setStyleSheet("""
                QPushButton {
                    color: white;
                    border: 2px solid white;
                    border-radius: 10px;
                    padding: 10px;
                    background-color: transparent;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 50);
                }
                QPushButton:pressed {
                    background-color: rgba(255, 255, 255, 100);
                }
            """)
            layout.addWidget(button)


    def start_camera(self):
        print("opening camera")
        self.right_layout.replaceWidget(self.canvas_label, self.video_widget)
        self.canvas_label.hide()
        self.camera.start()


    def stop_camera(self):
        self.camera.stop()

        self.right_layout.replaceWidget(self.video_widget, self.canvas_label)
        self.video_widget.hide()
        self.canvas_label.show()


    # TODO: implement these functions and add button state/style change
    def record_video(self):
        output_file = "output_video.mp4" 
        self.media_recorder.setOutputLocation(QUrl.fromLocalFile(os.path.abspath(output_file)))
        self.media_recorder.record() 

    def stop_recording(self):
        self.media_recorder.stop()


    def record_audio(self):
        audio_output_file = "output_audio.wav"
        self.audio_recorder.setOutputLocation(QUrl.fromLocalFile(os.path.abspath(audio_output_file)))
        self.audio_recorder.record()


    def stop_audio(self):
        self.audio_recorder.stop()


    def take_screenshot(self):
        self.image_capture.capture()
        self.image_capture.imageCaptured.connect(self.process_capture)


    def process_capture(self, requestId, image):
        file_name = f"screenshot_{requestId}.png"
        image.save(file_name) 
        print(f"Screenshot saved as {file_name}")


    def run_face_detection(self):
        self.start_camera()
        print("Running Face Detection")


    def run_gender_age_detection(self):
        self.start_camera()
        print("Running Gender/Age Detection")


    def run_object_detection(self):
        self.start_camera()
        print("Running Object Detection")


    def run_mask_detection(self):
        self.start_camera()
        print("Running Mask Detection")


    def choose_folder(self):
        self.input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        self.to_keep_directory = QFileDialog.getExistingDirectory(self, "Select 'To Keep' Directory")
        self.to_delete_directory = QFileDialog.getExistingDirectory(self, "Select 'To Delete' Directory")

        if self.input_directory and self.to_keep_directory and self.to_delete_directory:
            print(f"Selected directories:\nInput: {self.input_directory}\nTo Keep: {self.to_keep_directory}\nTo Delete: {self.to_delete_directory}")
    

    def create_folder_button(self):
        folder_button_widget = QWidget()
        folder_button_layout = QVBoxLayout()

        text_label = QLabel("Image")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("color: white;")
        folder_button_layout.addWidget(text_label)

        folder_button_widget.setLayout(folder_button_layout)
        folder_button_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {SECONDARY_COLOR};
                border: 2px solid white;
                border-radius: 10px;
                padding: 10px;
                min-width: 100px;
                min-height: 100px;
            }}
        """)

        return folder_button_widget


    def process_image_project(self):
        print("processing_image_project")


    def start_image_project(self):
        print("running image project")

        self.popup_window = QDialog(self)
        self.popup_window.setWindowTitle("Choose Folder")

        popup_width = int(self.width() * POPUP_SIZE_RATIO)
        popup_height = int(self.height() * POPUP_SIZE_RATIO)
        self.popup_window.setGeometry(100, 100, popup_width, popup_height)
        self.popup_window.setStyleSheet(f"background-color: {SECONDARY_COLOR};")

        popup_layout = QVBoxLayout()

        self.device_layout = QVBoxLayout()  
        cust_label = QLabel("Customization", self.popup_window)
        cust_label.setAlignment(Qt.AlignCenter)
        cust_label.setStyleSheet("color: white")
        popup_layout.addWidget(cust_label)
        popup_layout.addLayout(self.device_layout)

        folder_icon = QSvgWidget(UXUI_ASSETS + "./Assets_svg/ic_customization_upload_folder.svg")
        folder_icon.setFixedSize(100, 100)

        upload_icon = QSvgWidget(UXUI_ASSETS + "./Assets_svg/bt_function_upload_normal.svg")
        upload_icon.setFixedSize(40, 40)

        folder_button_widget = QWidget()
        folder_button_layout = QVBoxLayout()

        text_label = QLabel("Image")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("color: white; border: none")

        folder_button_layout.addWidget(text_label)
        folder_button_layout.addWidget(folder_icon)

        description_label = QLabel("Upload or drag files")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet("color: white; border: none")
        folder_button_layout.addWidget(description_label)
        folder_button_layout.addWidget(upload_icon)
        folder_button_layout.setAlignment(Qt.AlignCenter)

        folder_button_widget.setLayout(folder_button_layout)

        folder_frame = QFrame()
        folder_frame.setStyleSheet(f"""
            QFrame {{
                border: 2px solid white;
                border-radius: 10px;
                padding: 10px;
                background-color: {SECONDARY_COLOR};
            }}
        """)
        folder_frame.setLayout(QVBoxLayout())
        folder_frame.layout().addWidget(folder_button_widget)

        folder_icon2 = QSvgWidget(UXUI_ASSETS + "./Assets_svg/ic_customization_upload_folder.svg")
        folder_icon2.setFixedSize(100, 100)
        upload_icon2 = QSvgWidget(UXUI_ASSETS + "./Assets_svg/bt_function_upload_normal.svg")
        upload_icon2.setFixedSize(40, 40)
        folder_button_widget2 = QWidget()
        folder_button_layout2 = QVBoxLayout()

        text_label2 = QLabel("Label")
        text_label2.setAlignment(Qt.AlignCenter)
        text_label2.setStyleSheet("color: white; border: none")

        folder_button_layout2.addWidget(text_label2)
        folder_button_layout2.addWidget(folder_icon2)
        folder_button_layout2.setAlignment(Qt.AlignCenter)

        description_label2 = QLabel("Upload or drag files")
        description_label2.setAlignment(Qt.AlignCenter)
        description_label2.setStyleSheet("color: white; border: none")
        folder_button_layout2.addWidget(description_label2)
        folder_button_layout2.addWidget(upload_icon2)

        folder_button_widget2.setLayout(folder_button_layout2)

        folder_frame2 = QFrame()
        folder_frame2.setStyleSheet(f"""
            QFrame {{
                border: 2px solid white;
                border-radius: 10px;
                padding: 10px;
                background-color: {SECONDARY_COLOR};
            }}
        """)
        folder_frame2.setLayout(QVBoxLayout())
        folder_frame2.layout().addWidget(folder_button_widget2)

        folder_buttons_layout = QHBoxLayout()
        folder_buttons_layout.addWidget(folder_frame)
        folder_buttons_layout.addWidget(folder_frame2)

        popup_layout.addLayout(folder_buttons_layout)

        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton('Cancel', self.popup_window)
        self.cancel_button.clicked.connect(self.popup_window.close)
        self.cancel_button.setStyleSheet(BUTTON_STYLE)
        button_layout.addWidget(self.cancel_button)

        self.done_button = QPushButton('Done', self.popup_window)
        self.done_button.setStyleSheet(BUTTON_STYLE)
        self.done_button.clicked.connect(self.process_image_project)
        button_layout.addWidget(self.done_button) 

        popup_layout.addLayout(button_layout)

        self.popup_window.setLayout(popup_layout)

        self.popup_window.setModal(True)
        self.setEnabled(False)  
        self.popup_window.finished.connect(lambda: self.setEnabled(True))

        self.popup_window.show()


    def upload_model(self):
        print("Uploading Model")


    def main_page(self):
        self.clear_device_layout(self.layout)
        self.setWindowTitle('Innovedus AI Playground')
        self.setGeometry(100, 100, *WINDOW_SIZE)
        
        main_layout = QHBoxLayout()
        top_nav = QHBoxLayout()

        welcome_label = QLabel(self)
        welcome_pixmap = QPixmap(UXUI_ASSETS + "kneron_logo.png").scaled(150, 150, Qt.KeepAspectRatio)
        welcome_label.setPixmap(welcome_pixmap)
        top_nav.addWidget(welcome_label, alignment=Qt.AlignLeft)
        top_nav.addStretch()

        settings_button = QPushButton("Settings", self)
        settings_button.setStyleSheet(BUTTON_STYLE)
        top_nav.addWidget(settings_button, alignment=Qt.AlignRight)
        self.layout.addLayout(top_nav)

        
        left_widget = QWidget()
        left_widget.setLayout(self.left_layout)
        left_widget.setFixedWidth(self.geometry().width() // 3)

        right_widget = QWidget()
        right_widget.setLayout(self.right_layout)
        right_widget.setFixedWidth(self.geometry().width() * 2 // 3)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        self.layout.addLayout(main_layout)
        self.setLayout(self.layout)

        self.create_device_layout()
        self.create_right_layout()


    def show_device_details(self):
        print("show_device_details")
     
        
    def create_device_layout(self):
        devices_frame = self.create_frame("Device", UXUI_ASSETS + "./Assets_svg/ic_window_device.svg")
        devices_frame_layout = QVBoxLayout()
        self.device_list = QListWidget(self)

        print(self.connected_devices)

        for device in self.connected_devices:
            usb_port_id = device.get("usb_port_id")
            product_id = device.get("product_id")
            kn_number = device.get("kn_number")

            h_layout = QHBoxLayout()
            icon = QSvgWidget(UXUI_ASSETS + "./Assets_svg/ic_window_device.svg")
            icon.setFixedSize(40, 40)
            h_layout.addWidget(icon)

            text_layout = QVBoxLayout()
            line1_label = QLabel(f"Dongle: {product_id}")
            line1_label.setStyleSheet("font-weight: bold; color: white;")
            text_layout.addWidget(line1_label)

            line2_label = QLabel(f"KN number: {kn_number}")
            line2_label.setStyleSheet("color: white;")
            text_layout.addWidget(line2_label)

            h_layout.addLayout(text_layout)

            item_widget = QWidget()
            item_widget.setLayout(h_layout)
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.device_list.addItem(list_item)
            self.device_list.setItemWidget(list_item, item_widget)

        devices_frame_layout.addWidget(self.device_list)

        detail_button = QPushButton("Details", self)
        detail_button.clicked.connect(self.show_device_details)
        devices_frame_layout.addWidget(detail_button)

        devices_frame.setLayout(devices_frame_layout)
        self.left_layout.addWidget(devices_frame)

        self.models_frame = self.create_frame("AI Toolbox", UXUI_ASSETS + "./Assets_svg/ic_window_toolbox.svg")
        models_layout = QVBoxLayout(self.models_frame)
        self.models_frame.setLayout(models_layout)

        self.add_model_buttons(self.models_frame.layout())
        self.left_layout.addWidget(self.models_frame)


    def create_right_layout(self):
        self.canvas_label = QLabel("Canvas Area (Camera Screen)", self)
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setStyleSheet("border: 1px transparent; background: gray; border-radius: 20px; ")  
        self.right_layout.addWidget(self.canvas_label)

        button_overlay_layout = QVBoxLayout()
        button_overlay_layout.setContentsMargins(0, 0, 0, 0)

        self.create_square_buttons(button_overlay_layout)
        
        button_overlay_widget = QWidget(self)
        button_overlay_widget.setLayout(button_overlay_layout)
        button_overlay_widget.setStyleSheet(f"background: {SECONDARY_COLOR}; border-radius: 20px; padding: 10px;") 

        button_overlay_widget.setFixedHeight(150) 
        self.right_layout.addWidget(button_overlay_widget, alignment=Qt.AlignBottom | Qt.AlignRight)


    def create_square_buttons(self, layout):
        square_buttons_info = [
            ('video', UXUI_ASSETS + "./Assets_svg/ic_recording_camera.svg"),
            ('voice', UXUI_ASSETS + "./Assets_svg/ic_recording_voice.svg"),
            ('screenshot', UXUI_ASSETS + "./Assets_svg/bt_function_screencapture_normal.svg"),
        ]

        for button_name, icon_path in square_buttons_info:
            button = QPushButton(self)
            button.setFixedSize(50, 50)
            button.setStyleSheet(SQUARE_BUTTON_STYLE)
            button_layout = QHBoxLayout(button)
            button_layout.setContentsMargins(0, 0, 0, 0)
            icon = QSvgWidget(icon_path)
            icon.setFixedSize(40, 40)
            button_layout.addWidget(icon)

            layout.addWidget(button)


    def upload_model(self):
            model_file, _ = QFileDialog.getOpenFileName(self, "Upload Model", "", "NEF Files (*.nef)")
            if model_file:
                if model_file.endswith('.nef'):
                    model_name = os.path.basename(model_file)
                    self.model_buttons.insert(-1, (model_name, self.run_uploaded_model))
                    print(f"Model uploaded: {model_name}")
                    self.refresh_model_buttons()
                else:
                    self.show_error_popup("Invalid file format. Please upload a .nef file.")


    def refresh_model_buttons(self):
        layout = self.models_frame.layout()

        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()
        
        self.add_model_buttons(layout)


    def run_uploaded_model(self):
        print("Running uploaded model")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
