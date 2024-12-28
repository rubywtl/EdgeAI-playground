import os
import cv2
import kp
import numpy as np
import numpy as np
from sklearn.cluster import DBSCAN


# -------------------- General Dongle Connection -------------------------------#
def connect_and_load_firmware(device_descriptors):
    # device_descriptors = kp.core.scan_devices()
    if 0 < device_descriptors.device_descriptor_number:
        for device in device_descriptors.device_descriptor_list:
            usb_port_id = device.usb_port_id
            device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
            kp.core.set_timeout(device_group=device_group, milliseconds=5000)
            SCPU_FW_PATH = '../../res/firmware/KL520/fw_scpu.bin'
            NCPU_FW_PATH = '../../res/firmware/KL520/fw_ncpu.bin'
            kp.core.load_firmware_from_file(device_group=device_group,
                                            scpu_fw_path=SCPU_FW_PATH,
                                            ncpu_fw_path=NCPU_FW_PATH)
    else:
        print('Error: no Kneron device connect.')
        exit(0)


def list_image_files(directory):
    """List all image files in a given directory."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif') 
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]


# -------------------- Decluttering & Photo Quality Model -------------------------------#

def preprocess_image(image_file_path):
    maxbytes = 500000
    file_size = os.path.getsize(image_file_path)
    img = cv2.imread(filename=image_file_path)
    
    if file_size > maxbytes:
        scale_factor = (maxbytes / file_size) ** 0.5
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        if new_width % 2 != 0:
            new_width += 1
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        if img.shape[1] % 2 != 0:
            img = cv2.resize(img, (img.shape[1] + 1, img.shape[0]), interpolation=cv2.INTER_AREA)
    
    img_bgr565 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2BGR565)
    return img_bgr565


def perform_inference(device_group, model_nef_descriptor, img_bgr565):
    generic_inference_input_descriptor = kp.GenericImageInferenceDescriptor(
        model_id=model_nef_descriptor.models[0].id,
        inference_number=0,
        input_node_image_list=[
            kp.GenericInputNodeImage(
                image=img_bgr565,
                image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565,
                resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
                padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
            )
        ]
    )

    kp.inference.generic_image_inference_send(device_group=device_group,
                                              generic_inference_input_descriptor=generic_inference_input_descriptor)
    generic_raw_result = kp.inference.generic_image_inference_receive(device_group=device_group)

    inf_node_output_list = []

    for node_idx in range(generic_raw_result.header.num_output_node):
        inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=node_idx,
            generic_raw_result=generic_raw_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
        )
        inf_node_output_list.append(inference_float_node_output)

    return inf_node_output_list


def post_process_inference(inf_node_output_list):
    """Processes the inference output and returns a mean score."""
    data = inf_node_output_list[0]
    raw_ndarray = data.ndarray

    if isinstance(raw_ndarray, np.ndarray):
        ndarray_np = raw_ndarray
    else:
        ndarray_np = np.array(raw_ndarray)

    ndarray_np = ndarray_np.flatten()  
    ndarray_np = ndarray_np.reshape((data.channel, data.height, data.width))

    ndarray_np = ndarray_np.flatten()
    result = np.mean(ndarray_np)  # Mean score

    return result


def process_image(device_group, model_nef_descriptor, image_file_path):
    """Full pipeline: preprocess image, perform inference, and post-process to get a score."""
    img_bgr565 = preprocess_image(image_file_path)
    inf_node_output_list = perform_inference(device_group, model_nef_descriptor, img_bgr565)
    nd_array = inf_node_output_list[0].ndarray
    number = float(nd_array.flatten()[0])
    return number


def cosine_similarity(tensor1, tensor2):
    """Compute the cosine similarity between two tensors."""
    dot_product = np.dot(tensor1, tensor2)
    norm1 = np.linalg.norm(tensor1)
    norm2 = np.linalg.norm(tensor2)
    return dot_product / (norm1 * norm2)


def compare_images_cosine_similarity(image_paths, device_group, model_nef_descriptor):
    """Compare the cosine similarity between feature tensors of photos in the given image file paths."""
    num_images = len(image_paths)
    features = []
    for image_path in image_paths:
        feature = process_image(device_group, model_nef_descriptor, image_path)
        features.append(feature)
    
    similarity_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                similarity_matrix[i, j] = cosine_similarity(features[i], features[j])
            else:
                similarity_matrix[i, j] = 1.0
    return similarity_matrix


def cluster_images_with_dbscan(image_paths, feature_extractor, model_nef_descriptor, similarity_threshold=0.8, min_samples=2):
    """Cluster images based on cosine similarity using DBSCAN and return clusters as arrays of file paths."""
    similarity_matrix = compare_images_cosine_similarity(image_paths, feature_extractor, model_nef_descriptor)
    distance_matrix = 1 - similarity_matrix
    dbscan = DBSCAN(eps=1-similarity_threshold, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distance_matrix)
    
    clusters = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label != -1:
            cluster = [image_paths[i] for i in range(len(labels)) if labels[i] == label]
            clusters.append(cluster)
    return clusters



