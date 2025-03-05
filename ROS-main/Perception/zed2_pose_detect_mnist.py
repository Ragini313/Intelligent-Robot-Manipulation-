import cv2
import numpy as np
import tensorflow as tf

class CubeDetector:
    def __init__(self, model_path=None):
        # Load MNIST model for digit recognition if provided
        self.digit_model = None
        if model_path:
            try:
                self.digit_model = tf.keras.models.load_model(model_path)
                print("Digit recognition model loaded successfully")
            except Exception as e:
                print(f"Failed to load digit model: {e}")
    
    def process_images(self, rgb_image, depth_image=None):
        """Process RGB and depth images to detect cubes and digits"""
        try:
            # Make a copy of the image for visualization
            display_image = rgb_image.copy()
            
            # Detect cubes
            cube_positions = self.detect_cubes(rgb_image)
            results = []
            
            # Process each detected cube
            for i, (x, y, w, h) in enumerate(cube_positions):
                # Extract cube region
                cube_roi = rgb_image[y:y+h, x:x+w]
                
                # Get depth at cube center if depth data is available
                z = None
                if depth_image is not None:
                    center_y, center_x = y + h//2, x + w//2
                    if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                        z = depth_image[center_y, center_x]
                
                # Recognize digit on top face
                digit = self.recognize_digit(cube_roi)
                
                # Store results
                cube_info = {
                    'position': (x + w/2, y + h/2, z if z is not None else float('nan')),
                    'digit': digit,
                    'bbox': (x, y, w, h)
                }
                results.append(cube_info)
                
                # Visualize detection
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_image, f"Digit: {digit}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return results, display_image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return [], rgb_image
    
    def detect_cubes(self, image):
        """Detect cubes in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape to identify cubes
        cube_positions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 10000:  # Adjust based on your cube size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                # Square-ish shapes (aspect ratio close to 1)
                if 0.7 < aspect_ratio < 1.3:
                    cube_positions.append((x, y, w, h))
        
        return cube_positions
    
    def recognize_digit(self, cube_image):
        """Recognize the digit on top of the cube"""
        if self.digit_model is None:
            return "?"
            
        # Process for digit recognition:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(cube_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply thresholding to isolate the digit
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Find contours to locate the digit
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the digit)
            digit_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(digit_contour)
            
            # Extract digit region with some margin
            margin = 4  # pixels
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(cube_image.shape[1], x + w + margin)
            y_end = min(cube_image.shape[0], y + h + margin)
            
            digit_roi = thresh[y_start:y_end, x_start:x_end]
            
            # Resize to match input size expected by model (28x28 for MNIST)
            if digit_roi.size > 0:
                digit_roi = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
                
                # Normalize and prepare for model input
                digit_roi = digit_roi.astype('float32') / 255.0
                digit_roi = np.expand_dims(digit_roi, axis=0)
                digit_roi = np.expand_dims(digit_roi, axis=-1)
                
                # Predict using model
                prediction = self.digit_model.predict(digit_roi)
                return str(np.argmax(prediction[0]))
        
        return "?"

# Implementation

def main():
    # Path to your MNIST model
    model_path = None  # Replace with path to your model if available
    
    # Initialize detector
    detector = CubeDetector(model_path)
    
    # Load your RGB and depth images
    rgb_image = cv2.imread('/home/ragini/Desktop/iRobMan Lab/zed2_rgb_raw_image.png')
    depth_image = cv2.imread('/home/ragini/Desktop/iRobMan Lab/zed2_depth_image.png', cv2.IMREAD_ANYDEPTH)  # For 16-bit depth images
    
    # Process images
    results, display_image = detector.process_images(rgb_image, depth_image)
    
    # Print results
    for i, cube in enumerate(results):
        print(f"Cube {i+1}:")
        print(f"  Position (x, y, z): {cube['position']}")
        print(f"  Detected digit: {cube['digit']}")
        print(f"  Bounding box: {cube['bbox']}")
    
    # Display result
    cv2.imshow("Cube Detection", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



## Output is :
# Cube 1:
#   Position (x, y, z): (580.0, 294.5, 0)
#   Detected digit: ?
#   Bounding box: (567, 282, 26, 25)
# Cube 2:
#   Position (x, y, z): (37.5, 307.5, 35)
#   Detected digit: ?
#   Bounding box: (0, 272, 75, 71)




