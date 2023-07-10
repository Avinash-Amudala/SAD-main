from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import cv2
import numpy as np
from skimage import measure

app = Flask(__name__)

# Path to the model checkpoint
CHECKPOINT_PATH = "D:\SAD\sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

# Initialize the SAM
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
mask_generator = SamAutomaticMaskGenerator(sam)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file_path = os.path.join('static', filename)
        file.save(file_path)

        # Load the image
        image = cv2.imread(file_path)

        # Resize the image
        max_size = 500
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Generate masks for the image
        masks = mask_generator.generate(image)

        # Create an empty mask to accumulate all the object masks
        total_mask = np.zeros_like(image, dtype=np.uint8)

        # Apply the masks to the image
        for i, mask in enumerate(masks):
            binary_mask = mask['segmentation']  # Get the binary mask from the dictionary

            # Create a color for this object
            color = np.array([i % 256, (i * 50) % 256, (i * 100) % 256], dtype=np.uint8)

            # Apply the mask to the total mask
            for j in range(3):  # Apply the mask to each channel separately
                total_mask[binary_mask, j] = color[j]

            # Draw boundaries
            contours = measure.find_contours(binary_mask, 0.5)
            for contour in contours:
                contour = contour.astype(int)
                total_mask[contour[:, 0], contour[:, 1]] = [255, 0, 0]  # blue color for boundaries

        # Blend the original image with the total mask
        output = cv2.addWeighted(image, 0.5, total_mask, 0.5, 0)

        # Save the output image
        output_path = os.path.join('static', 'output_' + filename)
        cv2.imwrite(output_path, output)

        return render_template('index.html', filename='output_' + filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


