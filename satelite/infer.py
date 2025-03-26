import json
import cv2
import os
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Path to the test images directory
test_images_path = '/Users/mr.nerdhuge/Downloads/sayzek_project/namesles/test-images'

# Path to the image ID mapping file
image_id_mapping_path = '/Users/mr.nerdhuge/Downloads/sayzek_project/satelite/image_file_name_to_image_id.json'

# Load the image file name to image ID mapping
with open(image_id_mapping_path, 'r') as f:
    image_file_name_to_image_id = json.load(f)

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create the model using pre-trained weights on COCO dataset
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.to(device)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

results = []

# Loop through each image in the test directory
for img_name in os.listdir(test_images_path):
    # Read the image using OpenCV
    image_path = os.path.join(test_images_path, img_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        continue

    # Convert BGR (OpenCV) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    image_tensor = transform(image)
    image_tensor = image_tensor.to(device)

    # Add a batch dimension
    inputs = [image_tensor]

    # Run inference
    with torch.no_grad():
        outputs = model(inputs)

    # Get the predictions for the current image
    output = outputs[0]

    # Extract boxes, labels, and scores
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    # Filter out low-confidence detections
    confidence_threshold = 0.5  # Adjust as needed
    high_conf_indices = scores >= confidence_threshold
    boxes = boxes[high_conf_indices]
    labels = labels[high_conf_indices]
    scores = scores[high_conf_indices]

    # Map image file name to image ID
    img_id = image_file_name_to_image_id[img_name]

    # Process each detection
    for bbox, label, score in zip(boxes, labels, scores):
        # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Create the result dictionary
        res = {
            'image_id': img_id,
            'category_id': int(label),  # COCO dataset labels
            'bbox': [float(x_min), float(y_min), float(width), float(height)],
            'score': float("{:.8f}".format(score))
        }
        results.append(res)

# Save the results to a JSON file
with open('your_name.json', 'w') as f:
    json.dump(results, f)

print(f"Inference completed. Results saved to 'your_name.json'.")
