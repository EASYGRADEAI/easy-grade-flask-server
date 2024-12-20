import os
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
import json
from pathlib import Path

class AnswerClusterManager:
    def __init__(self, base_dir='answer_clusters', num_groups=4):
        """
        Initialize Answer Cluster Manager
        
        Args:
            base_dir (str): Base directory to store cluster models
            num_groups (int): Number of answer groups to create
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_groups = num_groups
        self.feature_extractor = cv2.SIFT_create()

    def extract_features(self, image):
        """
        Extract robust features from an answer image
        
        Args:
            image (np.ndarray): Answer image
        
        Returns:
            np.ndarray: Feature vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.feature_extractor.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(128)  # Default zero vector
        
        # Aggregate features (mean of descriptors)
        return np.mean(descriptors, axis=0)

    def get_question_cluster_path(self, question_id):
        """
        Generate path for a specific question's cluster model
        
        Args:
            question_id (str): Unique identifier for the question
        
        Returns:
            Path: Path to the cluster model file
        """
        return self.base_dir / f"question_{question_id}_clustering.joblib"

    def train_or_update_clustering(self, question_id, answer_images):
        """
        Train or update clustering for a specific question
        
        Args:
            question_id (str): Unique identifier for the question
            answer_images (List[np.ndarray]): List of answer images
        
        Returns:
            Dict: Clustering results
        """
        # Extract features from all answers
        features = [self.extract_features(img) for img in answer_images]
        features_array = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        
        # Determine optimal number of clusters
        num_clusters = min(self.num_groups, len(answer_images))
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=num_clusters, 
            random_state=42, 
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(normalized_features)
        
        # Save clustering model
        cluster_model_path = self.get_question_cluster_path(question_id)
        joblib.dump({
            'kmeans': kmeans,
            'scaler': scaler
        }, cluster_model_path)
        
        # Analyze cluster characteristics
        cluster_details = self._analyze_clusters(normalized_features, cluster_labels)
        
        return {
            'labels': cluster_labels.tolist(),
            'cluster_details': cluster_details,
            'model_path': str(cluster_model_path)
        }

    def _analyze_clusters(self, features, labels):
        """
        Analyze characteristics of clusters
        
        Args:
            features (np.ndarray): Normalized feature vectors
            labels (np.ndarray): Cluster labels
        
        Returns:
            Dict: Cluster analysis details
        """
        cluster_details = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_features = features[cluster_mask]
            
            cluster_details[int(label)] = {
                'size': len(cluster_features),
                'centroid': np.mean(cluster_features, axis=0).tolist(),
                'variance': np.var(cluster_features, axis=0).tolist()
            }
        
        return cluster_details

    def predict_cluster(self, question_id, new_answer_image):
        """
        Predict cluster for a new answer to a specific question
        
        Args:
            question_id (str): Unique identifier for the question
            new_answer_image (np.ndarray): New answer image
        
        Returns:
            int: Predicted cluster label
        """
        cluster_model_path = self.get_question_cluster_path(question_id)
        
        if not cluster_model_path.exists():
            raise ValueError(f"No clustering model found for question {question_id}")
        
        # Load saved model
        saved_model = joblib.load(cluster_model_path)
        kmeans = saved_model['kmeans']
        scaler = saved_model['scaler']
        
        # Extract and normalize features
        features = self.extract_features(new_answer_image)
        normalized_features = scaler.transform(features.reshape(1, -1))
        
        # Predict cluster
        return int(kmeans.predict(normalized_features)[0])

    def update_clustering_with_new_submissions(self, question_id, new_answer_images):
        """
        Update existing clustering model with new submissions
        
        Args:
            question_id (str): Unique identifier for the question
            new_answer_images (List[np.ndarray]): New answer images
        
        Returns:
            Dict: Updated clustering results
        """
        cluster_model_path = self.get_question_cluster_path(question_id)
        
        if not cluster_model_path.exists():
            # If no existing model, create a new one
            return self.train_or_update_clustering(question_id, new_answer_images)
        
        # Load existing model
        saved_model = joblib.load(cluster_model_path)
        kmeans = saved_model['kmeans']
        scaler = saved_model['scaler']
        
        # Extract features from new answers
        new_features = [self.extract_features(img) for img in new_answer_images]
        new_features_array = np.array(new_features)
        
        # Normalize new features using existing scaler
        normalized_new_features = scaler.transform(new_features_array)
        
        # Add new samples to the existing clustering
        updated_features = np.vstack([
            scaler.inverse_transform(kmeans.cluster_centers_), 
            new_features_array
        ])
        
        # Retrain clustering with all features
        updated_kmeans = KMeans(
            n_clusters=min(self.num_groups, len(updated_features)), 
            random_state=42, 
            n_init=10
        )
        updated_labels = updated_kmeans.fit_predict(updated_features)
        
        # Save updated model
        joblib.dump({
            'kmeans': updated_kmeans,
            'scaler': scaler
        }, cluster_model_path)
        
        # Analyze updated clusters
        cluster_details = self._analyze_clusters(
            scaler.transform(updated_features), 
            updated_labels
        )
        
        return {
            'labels': updated_labels.tolist(),
            'cluster_details': cluster_details,
            'model_path': str(cluster_model_path)
        }

# Example usage
if __name__ == "__main__":
    # Example of how to use the clustering system
    cluster_manager = AnswerClusterManager()
    
    # Simulate first batch of answers for a question
    initial_answers = [
        cv2.imread('answer1.jpg'),
        cv2.imread('answer2.jpg'),
        cv2.imread('answer3.jpg')
    ]
    
    # Train initial clustering
    initial_clustering = cluster_manager.train_or_update_clustering(
        question_id='q1', 
        answer_images=initial_answers
    )
    print("Initial Clustering:", initial_clustering)
    
    # Predict cluster for a new answer
    new_answer = cv2.imread('new_answer.jpg')
    predicted_cluster = cluster_manager.predict_cluster('q1', new_answer)
    print("Predicted Cluster:", predicted_cluster)
    
    # Update clustering with new submissions
    new_answers = [
        cv2.imread('additional_answer1.jpg'),
        cv2.imread('additional_answer2.jpg')
    ]
    updated_clustering = cluster_manager.update_clustering_with_new_submissions(
        question_id='q1', 
        new_answer_images=new_answers
    )
    print("Updated Clustering:", updated_clustering)