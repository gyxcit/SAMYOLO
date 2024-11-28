from flask import Flask, request, jsonify, render_template, url_for
from flask_socketio import SocketIO
import threading
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
from segment_anything import sam_model_registry, SamPredictor
import os
from werkzeug.utils import secure_filename
import logging
import json
import shutil
import sys
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
app = Flask(__name__)
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    SAM_RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'sam','sam_results')
    YOLO_RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'yolo','yolo_results')
    YOLO_TRAIN_IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'yolo','dataset_yolo','train','images')
    YOLO_TRAIN_LABEL_FOLDER = os.path.join(BASE_DIR, 'static', 'yolo','dataset_yolo','train','labels')
    AREA_DATA_FOLDER = os.path.join(BASE_DIR, 'static', 'yolo','area_data')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    SAM_CHECKPOINT = os.path.join(BASE_DIR, 'static', 'sam',"sam_vit_h_4b8939.pth")
    SAM_2 = os.path.join(BASE_DIR, 'static', 'sam',"sam2.1_hiera_tiny.pt")
    YOLO_PATH = os.path.join(BASE_DIR, 'static', 'yolo', "model_yolo.pt")
    RETRAINED_MODEL_PATH = os.path.join(BASE_DIR, 'static', 'yolo', "model_retrained.pt")
    DATA_PATH = os.path.join(BASE_DIR, 'static', 'yolo','dataset_yolo', "data.yaml")

app.config.from_object(Config)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAM_RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['YOLO_RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['YOLO_TRAIN_IMAGE_FOLDER'], exist_ok=True)
os.makedirs(app.config['YOLO_TRAIN_LABEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['AREA_DATA_FOLDER'], exist_ok=True)


# Initialize Yolo model
try:
    model = YOLO(app.config['YOLO_PATH'])
except Exception as e:
    logger.error(f"Failed to initialize YOLO model: {str(e)}")
    raise

try:
    sam2_checkpoint = app.config['SAM_2']
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    predictor = SAM2ImagePredictor(sam2_model)
except Exception as e:
    logger.error(f"Failed to initialize SAM model: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def scale_coordinates(coords, original_dims, target_dims):
    """
    Scale coordinates from one dimension space to another.
    
    Args:
        coords: List of [x, y] coordinates
        original_dims: Tuple of (width, height) of original space
        target_dims: Tuple of (width, height) of target space
    
    Returns:
        Scaled coordinates
    """
    scale_x = target_dims[0] / original_dims[0]
    scale_y = target_dims[1] / original_dims[1]
    
    return [
        [int(coord[0] * scale_x), int(coord[1] * scale_y)]
        for coord in coords
    ]

def scale_box(box, original_dims, target_dims):
    """
    Scale bounding box coordinates from one dimension space to another.
    
    Args:
        box: List of [x1, y1, x2, y2] coordinates
        original_dims: Tuple of (width, height) of original space
        target_dims: Tuple of (width, height) of target space
    
    Returns:
        Scaled box coordinates
    """
    scale_x = target_dims[0] / original_dims[0]
    scale_y = target_dims[1] / original_dims[1]
    
    return [
        int(box[0] * scale_x),  # x1
        int(box[1] * scale_y),  # y1
        int(box[2] * scale_x),  # x2
        int(box[3] * scale_y)   # y2
    ]

def retrain_model_fn():
    # Parameters for retraining
    data_path = app.config['DATA_PATH']
    epochs = 5
    img_size = 640
    batch_size = 8

    # Start training with YOLO, using event listeners for epoch completion
    for epoch in range(epochs):
        # Train the model for one epoch, here we simulate with a loop
        model.train(
            data=data_path,
            epochs=1,  # Use 1 epoch per call to get individual progress
            imgsz=img_size,
            batch=batch_size,
            device="cpu"  # Adjust based on system capabilities
        )

        # Emit an update to the client after each epoch
        socketio.emit('training_update', {
            'epoch': epoch + 1,
            'status': f"Epoch {epoch + 1} complete"
        })

    # Emit a message once training is complete
    socketio.emit('training_complete', {'status': "Retraining complete"})
    model.save(app.config['YOLO_PATH'])
    logger.info("Model retrained successfully")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yolo')
def yolo():
    return render_template('yolo.html')

@app.route('/upload_sam', methods=['POST'])
def upload_sam_file():
    """
    Handles SAM image upload and embeds the image into the predictor instance.

    Returns:
        JSON response with 'message', 'image_url', 'filename', and 'dimensions' keys
            on success, or 'error' key with an appropriate error message on failure.
    """

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Set the image for predictor right after upload
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Failed to load uploaded image'}), 500
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        logger.info("Image embedded successfully")
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        image_url = url_for('static', filename=f'uploads/{filename}')
        logger.info(f"File uploaded successfully: {filepath}")
        
        return jsonify({
            'message': 'File uploaded successfully',
            'image_url': image_url,
            'filename': filename,
            'dimensions': {
                'width': width,
                'height': height
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Server error during upload'}), 500

@app.route('/upload_yolo', methods=['POST'])
def upload_yolo_file():
    """
    Upload a YOLO image file
    
    This endpoint allows a POST request containing a single image file. The file is
    saved to the uploads folder and the image is embedded into the YOLO model.
    
    Returns a JSON response with the following keys:
    - message: a success message
    - image_url: the URL of the uploaded image
    - filename: the name of the uploaded file
    
    If an error occurs, the JSON response will contain an 'error' key with a
    descriptive error message.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        
        image_url = url_for('static', filename=f'uploads/{filename}')
        logger.info(f"File uploaded successfully: {filepath}")
        
        return jsonify({
            'message': 'File uploaded successfully',
            'image_url': image_url,
            'filename': filename,
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Server error during upload'}), 500

@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    """
    Generate a mask for a given image using the YOLO model
    @param data: a JSON object containing the following keys:
        - filename: the name of the image file
        - normalized_void_points: a list of normalized 2D points (x, y) representing the voids
        - normalized_component_boxes: a list of normalized 2D bounding boxes (x, y, w, h) representing the components
    @return: a JSON object containing the following keys:
        - status: a string indicating the status of the request
        - train_image_url: the URL of the saved train image
        - result_path: the URL of the saved result image
    """
    try:
        data = request.json
        normalized_void_points = data.get('void_points', [])
        normalized_component_boxes = data.get('component_boxes', [])
        filename = data.get('filename', '')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 500
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        
        # Denormalize coordinates back to pixel values
        void_points = [
            [int(point[0] * image_width), int(point[1] * image_height)]
            for point in normalized_void_points
        ]
        logger.info(f"Void points: {void_points}")
        
        component_boxes = [
            [
                int(box[0] * image_width),
                int(box[1] * image_height),
                int(box[2] * image_width),
                int(box[3] * image_height)
            ]
            for box in normalized_component_boxes
        ]
        logger.info(f"Void points: {void_points}")

        # Create a list to store individual void masks
        void_masks = []
        
        # Process void points one by one
        for point in void_points:
            # Convert point to correct format: [N, 2] array
            point_coord = np.array([[point[0], point[1]]])
            point_label = np.array([1])  # Single label
            
            masks, scores, _ = predictor.predict(
                point_coords=point_coord,
                point_labels=point_label,
                multimask_output=True  # Get multiple masks
            )
            
            if len(masks) > 0:  # Check if any masks were generated
                # Get the mask with highest score
                best_mask_idx = np.argmax(scores)
                void_masks.append(masks[best_mask_idx])
                logger.info(f"Processed void point {point} with score {scores[best_mask_idx]}")

        # Process component boxes
        component_masks = []
        if component_boxes:
            for box in component_boxes:
                # Convert box to correct format: [2, 2] array
                box_np = np.array([[box[0], box[1]], [box[2], box[3]]])
                masks, scores, _ = predictor.predict(
                    box=box_np,
                    multimask_output=True
                )
                if len(masks) > 0:
                    best_mask_idx = np.argmax(scores)
                    component_masks.append(masks[best_mask_idx])
                    logger.info(f"Processed component box {box}")

        # Create visualization with different colors for each void
        combined_image = image.copy()

        # Font settings for labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0,0,0)  # White text color
        font_thickness = 1
        background_color = (255, 255, 255)  # White background for text

        # Helper function to get bounding box coordinates
        def get_bounding_box(mask):
            coords = np.column_stack(np.where(mask))
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            return (x_min, y_min, x_max, y_max)
        
        # Helper function to add text with background
        def put_text_with_background(img, text, pos):
            # Calculate text size
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            # Define the rectangle coordinates for background
            background_tl = (pos[0], pos[1] - text_h - 2)
            background_br = (pos[0] + text_w, pos[1] + 2)
            # Draw white rectangle as background
            cv2.rectangle(img, background_tl, background_br, background_color, -1)
            # Put the text over the background rectangle
            cv2.putText(img, text, pos, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        def get_safe_label_position(x_min, y_min, x_max, y_max, text_w, text_h, img_width, img_height):
            # Default to top-right of bounding box
            x_pos = min(y_max, img_width - text_w - 10)  # Keep 10px margin from the right
            y_pos = max(x_min + text_h + 5, text_h + 5)  # Keep 5px margin from the top
            return x_pos, y_pos


        # Apply void masks with different colors
        for mask in void_masks:
            mask = mask.astype(bool)
            combined_image[mask, 0] = np.clip(0.5 * image[mask, 0] + 0.5 * 255, 0, 255)  # Red channel with transparency
            combined_image[mask, 1] = np.clip(0.5 * image[mask, 1], 0, 255)              # Green channel reduced
            combined_image[mask, 2] = np.clip(0.5 * image[mask, 2], 0, 255)
            logger.info("Mask Drawn")  

        # Apply component masks in green
        for mask in component_masks:
            mask = mask.astype(bool)
        # Only apply green where there is no red overlay
            non_red_area = mask & ~np.any([void_mask for void_mask in void_masks], axis=0)
            combined_image[non_red_area, 0] = np.clip(0.5 * image[non_red_area, 0], 0, 255)              # Reduced red channel
            combined_image[non_red_area, 1] = np.clip(0.5 * image[non_red_area, 1] + 0.5 * 255, 0, 255)  # Green channel
            combined_image[non_red_area, 2] = np.clip(0.5 * image[non_red_area, 2], 0, 255)
            logger.info("Mask Drawn") 


        # Add labels on top of masks
        for i,mask in enumerate(void_masks):
            x_min, y_min, x_max, y_max = get_bounding_box(mask)
            (text_w, text_h), _ = cv2.getTextSize("Void", font, font_scale, font_thickness)
            label_position = get_safe_label_position(x_min, y_min, x_max, y_max, text_w, text_h, combined_image.shape[1], combined_image.shape[0])
            put_text_with_background(combined_image, f"Void {i+1}", label_position)    

        for i,mask in enumerate(component_masks):
            i=i+1
            x_min, y_min, x_max, y_max = get_bounding_box(mask)
            (text_w, text_h), _ = cv2.getTextSize("Component", font, font_scale, font_thickness)
            label_position = get_safe_label_position(x_min, y_min, x_max, y_max, text_w, text_h, combined_image.shape[1], combined_image.shape[0])
            put_text_with_background(combined_image, f"Component {i}", label_position)

        # Prepare an empty list to store the output in the required format
        mask_coordinates = []

        for mask in void_masks:
            # Get contours from the mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Image dimensions
            height, width = mask.shape

            # For each contour, extract the normalized coordinates
            for contour in contours:
                contour_points = contour.reshape(-1, 2)  # Flatten to (N, 2) where N is the number of points
                normalized_points = contour_points / [width, height]  # Normalize to (0, 1)

                class_id = 1  # 1 for voids
                row = [class_id] + normalized_points.flatten().tolist()  # Flatten and add the class
                mask_coordinates.append(row)

        for mask in component_masks:
            # Get contours from the mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter to keep only the largest contour
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = [contours[0]] if contours else []
            # Image dimensions
            height, width = mask.shape

            # For each contour, extract the normalized coordinates
            for contour in largest_contour:
                contour_points = contour.reshape(-1, 2)  # Flatten to (N, 2) where N is the number of points
                normalized_points = contour_points / [width, height]  # Normalize to (0, 1)

                class_id = 0  # for components 
                row = [class_id] + normalized_points.flatten().tolist()  # Flatten and add the class
                mask_coordinates.append(row)

        mask_coordinates_filename = f'{filename}.txt'  # Create a unique filename
        mask_coordinates_path = os.path.join(app.config['YOLO_TRAIN_LABEL_FOLDER'], mask_coordinates_filename)


        with open(mask_coordinates_path, "w") as file:
            for row in mask_coordinates:
                # Join elements of the row into a string with spaces in between and write to the file
                file.write(" ".join(map(str, row)) + "\n")

        # Save train image
        train_image_filepath = os.path.join(app.config['YOLO_TRAIN_IMAGE_FOLDER'], filename)
        shutil.copy(image_path, train_image_filepath)
        train_image_url = url_for('static', filename=f'yolo/dataset_yolo/train/images/{filename}')

        # Save result
        result_filename = f'segmented_{filename}'
        result_path = os.path.join(app.config['SAM_RESULT_FOLDER'], result_filename)
        plt.imsave(result_path, combined_image)
        logger.info("Mask generation completed successfully")
        
        return jsonify({
            'status': 'success',
            'train_image_url':train_image_url,
            'result_path': url_for('static', filename=f'sam/sam_results/{result_filename}')
        })

    except Exception as e:
        logger.error(f"Mask generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify an image and return the classification result, area data, and the annotated image.

    Request body should contain a JSON object with a single key 'filename' specifying the image file to be classified.

    Returns a JSON object with the following keys:

    - status: 'success' if the classification is successful, 'error' if there is an error.
    - result_path: URL of the annotated image.
    - area_data: a list of dictionaries containing the area and overlap statistics for each component.
    - area_data_path: URL of the JSON file containing the area data.

    If there is an error, returns a JSON object with a single key 'error' containing the error message.
    """

    try:
        data = request.json
        filename = data.get('filename', '')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 500 

        results = model(image)
        result = results[0]

        component_masks = []
        void_masks = []

        # Extract masks and labels from results
        for mask, label in zip(result.masks.data, result.boxes.cls):
            mask_array = mask.cpu().numpy().astype(bool)  # Convert to a binary mask (boolean array)
            if label == 1:  # Assuming label '1' represents void
                void_masks.append(mask_array)
            elif label == 0:  # Assuming label '0' represents component
                component_masks.append(mask_array)

        # Calculate area and overlap statistics
        area_data = []
        for i, component_mask in enumerate(component_masks):
            component_area = np.sum(component_mask).item()  # Total component area in pixels
            void_area_within_component = 0
            max_void_area_percentage = 0
            
            # Calculate overlap of each void mask with the component mask
            for void_mask in void_masks:
                overlap_area = np.sum(void_mask & component_mask).item()  # Overlapping area
                void_area_within_component += overlap_area
                void_area_percentage = (overlap_area / component_area) * 100 if component_area > 0 else 0
                max_void_area_percentage = max(max_void_area_percentage, void_area_percentage)
            
            # Append data for this component
            area_data.append({
                "Image": filename,
                'Component': f'Component {i+1}',
                'Area': component_area,
                'Void Area (pixels)': void_area_within_component,
                'Void Area %': void_area_within_component / component_area * 100 if component_area > 0 else 0,
                'Max Void Area %': max_void_area_percentage
            })

        area_data_filename = f'area_data_{filename.split("/")[-1]}.json'  # Create a unique filename
        area_data_path = os.path.join(app.config['AREA_DATA_FOLDER'], area_data_filename)

        with open(area_data_path, 'w') as json_file:
            json.dump(area_data, json_file, indent=4)

        annotated_image = result.plot() 

        output_filename = f'output_{filename}'
        output_image_path = os.path.join(app.config['YOLO_RESULT_FOLDER'], output_filename)
        plt.imsave(output_image_path, annotated_image) 
        logger.info("Classification completed successfully")  

        return jsonify({
            'status': 'success',
            'result_path': url_for('static', filename=f'yolo/yolo_results/{output_filename}'),
            'area_data': area_data,
            'area_data_path': url_for('static', filename=f'yolo/area_data/{area_data_filename}')
        })
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({'error': str(e)}), 500

retraining_status = {
    'status': 'idle',
    'progress': None,
    'message': None
}

@app.route('/start_retraining', methods=['GET', 'POST'])
def start_retraining():
    """
    Start the model retraining process.

    If the request is a POST, start the model retraining process in a separate thread.
    If the request is a GET, render the retraining page.

    Returns:
        A JSON response with the status of the retraining process, or a rendered HTML page.
    """
    if request.method == 'POST':
        # Reset status
        global retraining_status
        retraining_status['status'] = 'in_progress'
        retraining_status['progress'] = 'Initializing'
        
        # Start retraining in a separate thread
        threading.Thread(target=retrain_model_fn).start()
        return jsonify({'status': 'started'})
    else:
        # GET request - render the retraining page
        return render_template('retrain.html')

# Event handler for client connection
@socketio.on('connect')
def handle_connect():
    print('Client connected')


if __name__ == '__main__':
    app.run(port=5001, debug=True)
