"""Medical models for disease classification from chest X-rays."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import cv2
import joblib
import logging
from pathlib import Path
from ..utils import config

logger = logging.getLogger(__name__)

class ChestXRayDataset(Dataset):
    """Dataset class for chest X-ray images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # For sample data, create synthetic images
        if isinstance(self.image_paths[idx], str) and self.image_paths[idx].startswith('synthetic_'):
            # Create a synthetic chest X-ray-like image
            image = self.create_synthetic_xray(self.labels[idx])
        else:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]
    
    def create_synthetic_xray(self, label):
        """Create a synthetic chest X-ray image for demonstration."""
        # Create a 224x224 grayscale image that resembles a chest X-ray
        img = np.zeros((224, 224), dtype=np.uint8)
        
        # Add chest cavity shape
        cv2.ellipse(img, (112, 140), (80, 60), 0, 0, 360, 50, -1)
        
        # Add some noise based on the condition
        if label == 0:  # Normal
            noise = np.random.normal(40, 10, (224, 224))
        elif label == 1:  # Pneumonia
            noise = np.random.normal(60, 15, (224, 224))
            # Add some patches to simulate pneumonia
            cv2.circle(img, (90, 120), 20, 80, -1)
            cv2.circle(img, (140, 110), 15, 70, -1)
        elif label == 2:  # COVID-19
            noise = np.random.normal(55, 12, (224, 224))
            # Add ground glass opacities pattern
            for i in range(5):
                x, y = np.random.randint(50, 174, 2)
                cv2.circle(img, (x, y), np.random.randint(8, 15), 75, -1)
        else:  # Tuberculosis
            noise = np.random.normal(45, 8, (224, 224))
            # Add cavitation patterns
            cv2.circle(img, (100, 100), 12, 30, -1)
            cv2.circle(img, (130, 130), 8, 25, -1)
        
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        img_rgb = np.stack([img, img, img], axis=2)
        return Image.fromarray(img_rgb)

class ChestXRayClassifier:
    """High-precision chest X-ray disease classification model."""
    
    def __init__(self, model_type="resnet"):
        self.model_type = model_type
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = config.MEDICAL_CLASSES
        self.num_classes = len(self.classes)
        self.is_trained = False
        
        # Image transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_sample_data(self, n_samples=2000):
        """Create sample chest X-ray data for training."""
        np.random.seed(config.RANDOM_STATE)
        
        # Create synthetic image paths and labels
        image_paths = []
        labels = []
        
        samples_per_class = n_samples // self.num_classes
        
        for class_idx, class_name in enumerate(self.classes):
            for i in range(samples_per_class):
                image_paths.append(f"synthetic_{class_name}_{i}.jpg")
                labels.append(class_idx)
        
        # Shuffle the data
        indices = np.random.permutation(len(image_paths))
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        return image_paths, labels
    
    def create_cnn_model(self):
        """Create a custom CNN model."""
        class ChestXRayCNN(nn.Module):
            def __init__(self, num_classes):
                super(ChestXRayCNN, self).__init__()
                self.features = nn.Sequential(
                    # First conv block
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Second conv block
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Third conv block
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    
                    # Fourth conv block
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(512 * 7 * 7, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return ChestXRayCNN(self.num_classes)
    
    def create_resnet_model(self):
        """Create a ResNet-based model."""
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in list(model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        
        return model
    
    def extract_features_for_ml(self, image_paths, labels):
        """Extract features for traditional ML models."""
        features = []
        
        for img_path, label in zip(image_paths, labels):
            # Create synthetic image
            dataset = ChestXRayDataset([img_path], [label], transform=self.val_transform)
            img, _ = dataset[0]
            
            # Convert to numpy and extract features
            img_np = img.permute(1, 2, 0).numpy()
            img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Extract statistical features
            feature_vector = [
                np.mean(img_gray),
                np.std(img_gray),
                np.median(img_gray),
                np.min(img_gray),
                np.max(img_gray),
                np.percentile(img_gray, 25),
                np.percentile(img_gray, 75),
                cv2.Laplacian(img_gray, cv2.CV_64F).var(),  # Variance of Laplacian
            ]
            
            # Add histogram features
            hist = cv2.calcHist([img_gray], [0], None, [16], [0, 256])
            feature_vector.extend(hist.flatten() / img_gray.size)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_cnn(self, train_loader, val_loader, epochs=50):
        """Train a CNN model."""
        if self.model_type == "resnet":
            model = self.create_resnet_model()
        else:
            model = self.create_cnn_model()
        
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            scheduler.step()
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        self.model = model
        return best_val_acc / 100
    
    def train_traditional_ml(self, X_train, y_train, X_val, y_val):
        """Train traditional ML models on extracted features."""
        if self.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=config.RANDOM_STATE
            )
        elif self.model_type == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE
            )
        else:
            raise ValueError(f"Unknown traditional ML model type: {self.model_type}")
        
        model.fit(X_train, y_train)
        self.model = model
        
        val_accuracy = model.score(X_val, y_val)
        logger.info(f"{self.model_type} validation accuracy: {val_accuracy:.4f}")
        
        return val_accuracy
    
    def train(self, image_paths=None, labels=None):
        """Train the chest X-ray classification model."""
        if image_paths is None or labels is None:
            image_paths, labels = self.create_sample_data()
        
        logger.info(f"Training {self.model_type} model with {len(image_paths)} samples")
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=config.RANDOM_STATE,
            stratify=labels
        )
        
        if self.model_type in ["cnn", "resnet"]:
            # Create datasets and loaders for deep learning
            train_dataset = ChestXRayDataset(train_paths, train_labels, self.train_transform)
            val_dataset = ChestXRayDataset(val_paths, val_labels, self.val_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            accuracy = self.train_cnn(train_loader, val_loader)
        else:
            # Extract features for traditional ML
            logger.info("Extracting features for traditional ML...")
            X_train = self.extract_features_for_ml(train_paths, train_labels)
            X_val = self.extract_features_for_ml(val_paths, val_labels)
            
            accuracy = self.train_traditional_ml(X_train, train_labels, X_val, val_labels)
        
        self.is_trained = True
        
        if accuracy < config.TARGET_ACCURACY:
            logger.warning(f"Model accuracy {accuracy:.4f} is below target {config.TARGET_ACCURACY}")
        
        return {
            'validation_accuracy': accuracy,
            'model_type': self.model_type,
            'classes': self.classes
        }
    
    def predict_with_confidence(self, image_paths):
        """Make predictions with confidence scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        confidence_scores = []
        
        if self.model_type in ["cnn", "resnet"]:
            # Deep learning prediction
            self.model.eval()
            
            with torch.no_grad():
                for img_path in image_paths:
                    dataset = ChestXRayDataset([img_path], [0], self.val_transform)
                    image, _ = dataset[0]
                    image = image.unsqueeze(0).to(self.device)
                    
                    outputs = self.model(image)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    predictions.append(predicted_class)
                    confidence_scores.append(confidence)
        else:
            # Traditional ML prediction
            X = self.extract_features_for_ml(image_paths, [0] * len(image_paths))
            
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'classes': self.classes,
            'is_trained': self.is_trained
        }
        
        if self.model_type in ["cnn", "resnet"]:
            # Save PyTorch model state dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'classes': self.classes,
                'is_trained': self.is_trained
            }, filepath)
        else:
            # Save sklearn model
            joblib.dump(model_data, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        if self.model_type in ["cnn", "resnet"]:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            if self.model_type == "resnet":
                self.model = self.create_resnet_model()
            else:
                self.model = self.create_cnn_model()
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.classes = checkpoint['classes']
            self.is_trained = checkpoint['is_trained']
        else:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.classes = model_data['classes']
            self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")

def analyze_chest_xray(image_path, model):
    """Analyze a chest X-ray image and return diagnostic information."""
    if not model.is_trained:
        raise ValueError("Model must be trained before analysis")
    
    predictions, confidence_scores = model.predict_with_confidence([image_path])
    
    predicted_class = predictions[0]
    confidence = confidence_scores[0]
    
    diagnosis = model.classes[predicted_class]
    
    analysis = {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'recommendation': get_medical_recommendation(diagnosis, confidence),
        'requires_expert_review': confidence < config.MIN_CONFIDENCE_THRESHOLD
    }
    
    return analysis

def get_medical_recommendation(diagnosis, confidence):
    """Get medical recommendation based on diagnosis and confidence."""
    recommendations = {
        'normal': 'No immediate concerns detected. Continue routine screening.',
        'pneumonia': 'Possible pneumonia detected. Recommend immediate clinical evaluation and antibiotic treatment consideration.',
        'covid19': 'COVID-19 pattern detected. Recommend PCR testing, isolation, and clinical monitoring.',
        'tuberculosis': 'TB pattern detected. Recommend immediate sputum testing, contact tracing, and specialist referral.'
    }
    
    base_recommendation = recommendations.get(diagnosis, 'Unknown condition detected.')
    
    if confidence < config.MIN_CONFIDENCE_THRESHOLD:
        base_recommendation += f" LOW CONFIDENCE ({confidence:.1%}) - Urgent expert radiologist review required."
    elif confidence < 0.98:
        base_recommendation += f" Moderate confidence ({confidence:.1%}) - Consider expert review."
    
    return base_recommendation
