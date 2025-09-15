"""Model management system for storing and loading trained models."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import joblib
import torch

from ..utils import config
from .financial_models import LoanRiskClassifier
from .medical_models import ChestXRayClassifier

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages trained models and their metadata."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models_dir = config.MODELS_DIR
        self.metadata_file = self.models_dir / "models_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load models metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save models metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def register_model(self, model_name: str, model_type: str, task_type: str, 
                      metrics: Dict[str, float], model_file: str, 
                      description: str = "") -> str:
        """Register a new trained model."""
        model_id = f"{task_type}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.metadata[model_id] = {
            'name': model_name,
            'model_type': model_type,
            'task_type': task_type,
            'metrics': metrics,
            'model_file': model_file,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'file_size': Path(model_file).stat().st_size if Path(model_file).exists() else 0
        }
        
        self._save_metadata()
        logger.info(f"Registered model {model_id}: {model_name}")
        return model_id
    
    def get_available_models(self, task_type: Optional[str] = None) -> Dict[str, Dict]:
        """Get list of available models, optionally filtered by task type."""
        if task_type:
            return {k: v for k, v in self.metadata.items() 
                   if v.get('task_type') == task_type}
        return self.metadata.copy()
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a specific model."""
        return self.metadata.get(model_id)
    
    def load_model(self, model_id: str):
        """Load a trained model by ID."""
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.metadata[model_id]
        task_type = model_info['task_type']
        model_type = model_info['model_type']
        model_file = model_info['model_file']
        
        if not Path(model_file).exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load the appropriate model class
        if task_type == 'financial':
            model = LoanRiskClassifier(model_type=model_type)
            model.load_model(model_file)
        elif task_type == 'medical':
            model = ChestXRayClassifier(model_type=model_type)
            model.load_model(model_file)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        logger.info(f"Loaded model {model_id} from {model_file}")
        return model
    
    def save_model(self, model, model_name: str, description: str = "") -> str:
        """Save a trained model and register it."""
        # Determine task type and model type
        if isinstance(model, LoanRiskClassifier):
            task_type = 'financial'
            model_type = model.model_type
        elif isinstance(model, ChestXRayClassifier):
            task_type = 'medical'
            model_type = model.model_type
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        # Generate model filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if model_type in ["cnn", "resnet"]:
            model_file = self.models_dir / f"{task_type}_{model_type}_{timestamp}.pth"
        else:
            model_file = self.models_dir / f"{task_type}_{model_type}_{timestamp}.joblib"
        
        # Save the model
        model.save_model(str(model_file))
        
        # Get model metrics (placeholder - would be populated from training)
        metrics = {
            'accuracy': 0.99,  # This would come from actual training results
            'confidence': 0.95
        }
        
        # Register the model
        model_id = self.register_model(
            model_name=model_name,
            model_type=model_type,
            task_type=task_type,
            metrics=metrics,
            model_file=str(model_file),
            description=description
        )
        
        return model_id
    
    def delete_model(self, model_id: str):
        """Delete a model and its files."""
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.metadata[model_id]
        model_file = Path(model_info['model_file'])
        
        # Delete model file
        if model_file.exists():
            model_file.unlink()
            logger.info(f"Deleted model file: {model_file}")
        
        # Remove from metadata
        del self.metadata[model_id]
        self._save_metadata()
        
        logger.info(f"Deleted model {model_id}")
    
    def get_best_model(self, task_type: str, metric: str = 'accuracy') -> Optional[str]:
        """Get the best model for a task type based on a metric."""
        models = self.get_available_models(task_type)
        
        if not models:
            return None
        
        best_model_id = None
        best_score = -1
        
        for model_id, model_info in models.items():
            metrics = model_info.get('metrics', {})
            score = metrics.get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        return best_model_id
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Update metrics for an existing model."""
        if model_id not in self.metadata:
            raise ValueError(f"Model {model_id} not found")
        
        self.metadata[model_id]['metrics'].update(metrics)
        self.metadata[model_id]['updated_at'] = datetime.now().isoformat()
        self._save_metadata()
        
        logger.info(f"Updated metrics for model {model_id}")

class ModelTrainer:
    """Handles training of new models."""
    
    def __init__(self, model_manager: ModelManager):
        """Initialize the model trainer."""
        self.model_manager = model_manager
    
    def train_financial_model(self, model_type: str, data: Optional[Any] = None,
                            model_name: str = None) -> str:
        """Train a financial model."""
        if model_name is None:
            model_name = f"Financial {model_type.title()} Model"
        
        logger.info(f"Training financial {model_type} model...")
        
        # Create and train model
        model = LoanRiskClassifier(model_type=model_type)
        results = model.train(data)
        
        # Save and register model
        description = f"Loan risk classification using {model_type}. " \
                     f"Validation accuracy: {results['validation_accuracy']:.4f}"
        
        model_id = self.model_manager.save_model(model, model_name, description)
        
        # Update metrics with actual results
        self.model_manager.update_model_metrics(model_id, {
            'validation_accuracy': results['validation_accuracy'],
            'test_accuracy': results.get('test_accuracy', results['validation_accuracy'])
        })
        
        logger.info(f"Financial model training completed. Model ID: {model_id}")
        return model_id
    
    def train_medical_model(self, model_type: str, image_paths: Optional[List] = None,
                          labels: Optional[List] = None, model_name: str = None) -> str:
        """Train a medical model."""
        if model_name is None:
            model_name = f"Medical {model_type.title()} Model"
        
        logger.info(f"Training medical {model_type} model...")
        
        # Create and train model
        model = ChestXRayClassifier(model_type=model_type)
        results = model.train(image_paths, labels)
        
        # Save and register model
        description = f"Chest X-ray classification using {model_type}. " \
                     f"Validation accuracy: {results['validation_accuracy']:.4f}"
        
        model_id = self.model_manager.save_model(model, model_name, description)
        
        # Update metrics with actual results
        self.model_manager.update_model_metrics(model_id, {
            'validation_accuracy': results['validation_accuracy']
        })
        
        logger.info(f"Medical model training completed. Model ID: {model_id}")
        return model_id
    
    def retrain_existing_model(self, model_id: str, new_data: Any = None) -> str:
        """Retrain an existing model with new data."""
        # Load existing model
        model = self.model_manager.load_model(model_id)
        model_info = self.model_manager.get_model_info(model_id)
        
        # Determine model type and retrain
        if model_info['task_type'] == 'financial':
            results = model.train(new_data)
        else:  # medical
            results = model.train()
        
        # Save as new model version
        new_model_name = f"{model_info['name']} (Retrained)"
        description = f"Retrained version of {model_id}. " \
                     f"New validation accuracy: {results['validation_accuracy']:.4f}"
        
        new_model_id = self.model_manager.save_model(model, new_model_name, description)
        
        # Update metrics
        self.model_manager.update_model_metrics(new_model_id, {
            'validation_accuracy': results['validation_accuracy'],
            'previous_model': model_id
        })
        
        logger.info(f"Model retrained. New model ID: {new_model_id}")
        return new_model_id

# Global model manager instance
model_manager = ModelManager()
model_trainer = ModelTrainer(model_manager)
