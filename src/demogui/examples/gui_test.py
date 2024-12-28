import os
import kp
import shutil
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from demogui.utils import preprocess_image, perform_inference, post_process_inference, process_image, cosine_similarity, compare_images_cosine_similarity, cluster_images_with_dbscan, list_image_files


class ConsoleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demo Console")
        self.setGeometry(100, 100, 600, 400)

        # Console area for log messages
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        # Button to start the declutter process
        self.start_button = QPushButton("Select Directories and Start")
        self.start_button.clicked.connect(self.select_directories_and_start)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.start_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Variables to store directory paths
        self.input_directory = ''
        self.to_keep_directory = ''
        self.to_delete_directory = ''

    def print_message(self, message):
        self.text_edit.append(message)
        QApplication.processEvents()  # Update the GUI immediately

    def select_directories_and_start(self):
        # Open directory dialog to select directories
        self.input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        self.to_keep_directory = QFileDialog.getExistingDirectory(self, "Select 'To Keep' Directory")
        self.to_delete_directory = QFileDialog.getExistingDirectory(self, "Select 'To Delete' Directory")

        if self.input_directory and self.to_keep_directory and self.to_delete_directory:
            self.print_message(f"Selected directories:\nInput: {self.input_directory}\nTo Keep: {self.to_keep_directory}\nTo Delete: {self.to_delete_directory}")
            declutter_photo_album(self.input_directory, self.to_keep_directory, self.to_delete_directory, self)

def declutter_photo_album(input_directory, to_keep_directory, to_delete_directory, console):
    def log(message):
        console.print_message(message)

    log("CONNECTING DEVICE")

    DECLUTTER_MODEL_FILE_PATH = './resnet34_feature_extractor.nef'
    PHOTO_QUALITY_SCORER_PATH = './photo_scorer_520.nef'

    # Ensure output directories exist
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(to_keep_directory, exist_ok=True)
    os.makedirs(to_delete_directory, exist_ok=True)

    # Scan for devices
    device_descriptors = kp.core.scan_devices()
    if device_descriptors.device_descriptor_number > 0:
        usb_port_id = device_descriptors.device_descriptor_list[0].usb_port_id
        log(f"Device connected at USB port ID: {usb_port_id}")
    else:
        log('Error: no Kneron device connected.')
        return

    # Connect to device
    device_group = kp.core.connect_devices(usb_port_ids=[22])
    kp.core.set_timeout(device_group=device_group, milliseconds=5000)

    SCPU_FW_PATH = '../../res/firmware/KL520/fw_scpu.bin'
    NCPU_FW_PATH = '../../res/firmware/KL520/fw_ncpu.bin'
    kp.core.load_firmware_from_file(device_group=device_group,
                                    scpu_fw_path=SCPU_FW_PATH,
                                    ncpu_fw_path=NCPU_FW_PATH)

    # Filter low-quality images
    log("FILTERING LOW QUALITY IMAGES")
    model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                        file_path=PHOTO_QUALITY_SCORER_PATH)
    to_keep_images = []
    images = list_image_files(input_directory)
    for image_file_path in images:
        score = process_image(device_group, model_nef_descriptor, image_file_path)
        log(f"Image: {image_file_path}, Score: {score}")
        if score > 0.5:
            log("     Low quality: recommend to delete")
            shutil.copy(image_file_path, to_delete_directory)
        else:
            log("     Accepted quality image")
            to_keep_images.append(image_file_path)

    # Compare photo similarity
    log("COMPARING PHOTO SIMILARITY")
    model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                        file_path=DECLUTTER_MODEL_FILE_PATH)

    images = to_keep_images
    clusters = cluster_images_with_dbscan(images, device_group, model_nef_descriptor)
    
    # Organize clustered images into directories
    for cluster_index, cluster in enumerate(clusters):
        cluster_dir = os.path.join(to_delete_directory, f"cluster_{cluster_index}")
        os.makedirs(cluster_dir, exist_ok=True)
        log(f"Cluster #{cluster_index}")
        for image_file_path in cluster:
            log(image_file_path)
            shutil.copy(image_file_path, cluster_dir)

    # Move images not in any cluster to 'to_keep' directory
    non_clustered_images = set(images) - set(img for cluster in clusters for img in cluster)
    for image_file_path in non_clustered_images:
        shutil.copy(image_file_path, to_keep_directory)

    # Decide which photos to keep in clusters
    log("DECIDING WHICH PHOTO TO KEEP")
    for cluster_index, cluster in enumerate(clusters):
        cluster_scores = []
        for image_file_path in cluster:
            score = process_image(device_group, model_nef_descriptor, image_file_path)
            cluster_scores.append((image_file_path, score))

        # Sort by score in descending order and keep the top 2
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        top_photos = cluster_scores[:2]

        log(f"In cluster {cluster_index}")
        for image_file_path, _ in top_photos:
            log(f"Keep image: {image_file_path}")
            shutil.copy(image_file_path, to_keep_directory)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    console = ConsoleWindow()
    console.show()
    sys.exit(app.exec_())
