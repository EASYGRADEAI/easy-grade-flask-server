import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import logging
from pathlib import Path

class AdvancedAnswerSheetProcessor:
    def __init__(self, 
                 model_path=None, 
                 confidence_threshold=0.5, 
                 min_box_area=500):
        """
        Initialize the Answer Sheet Processor with YOLO object detection
        
        Args:
            model_path (str, optional): Path to custom trained YOLO model
            confidence_threshold (float): Minimum confidence for object detection
            min_box_area (int): Minimum pixel area for a detected box
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load YOLO model
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # Use pre-trained YOLO model for general object detection
                self.model = YOLO('yolov8n.pt')  # Nano version for faster inference
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
        
        # Configuration parameters
        self.confidence_threshold = confidence_threshold
        self.min_box_area = min_box_area

    def preprocess_image(self, image):
        """
        Preprocess image for consistent processing
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize image if too large
        height, width = image.shape[:2]
        max_dimension = 1024
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        
        # Optional: Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        processed = cv2.merge((l2,a,b))
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        return processed

    def detect_answer_boxes(self, image):
        """
        Detect potential answer boxes in an image
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            List of detected answer box coordinates
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run YOLO detection
        try:
            # Run inference (adjust classes as needed)
            results = self.model(processed_image, 
                                 conf=self.confidence_threshold, 
                                 classes=[0])  # 0 is typically 'person', modify as needed
            
            # Process and filter detections
            boxes = []
            for result in results:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Calculate box area and filter
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > self.min_box_area:
                        boxes.append([x1, y1, x2, y2])
            
            return boxes
        
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def extract_answer_sections(self, image, boxes):
        """
        Extract answer sections from detected boxes
        
        Args:
            image (np.ndarray): Source image
            boxes (List[List[int]]): Detected box coordinates
        
        Returns:
            List of extracted answer section images
        """
        # Sort boxes from top to bottom
        sorted_boxes = sorted(boxes, key=lambda x: x[1])
        
        answer_sections = []
        for box in sorted_boxes:
            x1, y1, x2, y2 = box
            # Ensure box is within image bounds
            y1 = max(0, y1)
            x1 = max(0, x1)
            y2 = min(image.shape[0], y2)
            x2 = min(image.shape[1], x2)
            
            answer_section = image[y1:y2, x1:x2]
            answer_sections.append(answer_section)
        
        return answer_sections

    def cluster_answers(self, answer_sections, n_clusters=3):
        """
        Cluster answer sections based on visual features
        
        Args:
            answer_sections (List[np.ndarray]): List of answer section images
            n_clusters (int): Number of clusters to create
        
        Returns:
            Cluster labels for each answer section
        """
        # Use SIFT for more robust feature extraction
        sift = cv2.SIFT_create()
        feature_vectors = []
        
        for section in answer_sections:
            # Convert to grayscale
            gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            # Aggregate features
            if descriptors is not None and len(descriptors) > 0:
                # Use mean of descriptors
                feature_vec = np.mean(descriptors, axis=0)
            else:
                # Fallback to zero vector if no features
                feature_vec = np.zeros(128)  # SIFT descriptor length
            
            feature_vectors.append(feature_vec)
        
        # Perform clustering
        if len(feature_vectors) > 0:
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(feature_vectors)), 
                random_state=42, 
                n_init=10
            )
            labels = kmeans.fit_predict(feature_vectors)
            return labels
        
        return np.array([])

    def process_submission(self, image_path, save_visualization=False):
        """
        Process a single answer sheet submission
        
        Args:
            image_path (str): Path to the image file
            save_visualization (bool): Whether to save visualization of detections
        
        Returns:
            Dict containing processing results
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Detect answer boxes
            boxes = self.detect_answer_boxes(image)
            
            if not boxes:
                self.logger.warning("No answer boxes detected")
                return {'status': 'warning', 'message': 'No answer boxes found'}
            
            # Extract answer sections
            answer_sections = self.extract_answer_sections(image, boxes)
            
            # Cluster answers
            cluster_labels = self.cluster_answers(answer_sections)
            
            # Optional: Visualize detections
            if save_visualization:
                self._visualize_detections(image, boxes, cluster_labels)
            
            return {
                'status': 'success',
                'total_answers': len(answer_sections),
                'clusters': cluster_labels.tolist(),
                'num_clusters': len(set(cluster_labels))
            }
        
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            return {'status': 'error', 'message': str(e)}

    def _visualize_detections(self, image, boxes, cluster_labels):
        """
        Create a visualization of detected boxes and clusters
        
        Args:
            image (np.ndarray): Original image
            boxes (List[List[int]]): Detected box coordinates
            cluster_labels (np.ndarray): Cluster labels
        """
        # Create a copy for drawing
        vis_image = image.copy()
        
        # Color palette for clusters
        colors = [
            (255,0,0),  # Blue
            (0,255,0),  # Green
            (0,0,255),  # Red
            (255,255,0),  # Cyan
            (255,0,255)  # Magenta
        ]
        
        # Draw boxes with cluster colors
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Select color based on cluster (or default to blue)
            color = colors[cluster_labels[i] % len(colors)] if i < len(cluster_labels) else (255,0,0)
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Optionally add cluster label
            cv2.putText(vis_image, 
                        f"Cluster {cluster_labels[i] if i < len(cluster_labels) else 'N/A'}", 
                        (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2)
        
        # Save visualization
        output_path = 'detection_visualization.jpg'
        cv2.imwrite(output_path, vis_image)
        self.logger.info(f"Visualization saved to {output_path}")

# Example usage
if __name__ == "__main__":
    processor = AdvancedAnswerSheetProcessor()
    result = processor.process_submission("sample_answer_sheet.jpg", save_visualization=True)
    print(result)