"""Financial models for loan underwriting and document analysis."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path
import logging
from ..utils import config

logger = logging.getLogger(__name__)

class LoanRiskClassifier:
    """High-precision loan risk classification model."""
    
    def __init__(self, model_type="random_forest"):
        """Initialize the loan risk classifier."""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = config.FINANCIAL_FEATURES
        self.is_trained = False
        
    def create_sample_data(self, n_samples=10000):
        """Create realistic sample loan data for training."""
        np.random.seed(config.RANDOM_STATE)
        
        # Generate realistic loan data
        data = {
            'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
            'income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 500000),
            'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 0.8,
            'loan_amount': np.random.lognormal(10, 0.8, n_samples).clip(5000, 1000000),
            'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
            'home_ownership': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),  # rent, own, mortgage
            'loan_purpose': np.random.choice(range(7), n_samples),  # various purposes
            'annual_income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 500000)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable with realistic risk assessment
        risk_score = (
            (df['credit_score'] - 600) / 250 * 0.4 +
            (df['income'] / 100000) * 0.2 +
            (1 - df['debt_to_income_ratio']) * 0.3 +
            (df['employment_length'] / 20) * 0.1
        )
        
        # Add some noise and create binary classification
        risk_score += np.random.normal(0, 0.1, n_samples)
        df['loan_approved'] = (risk_score > 0.5).astype(int)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the loan data."""
        X = df[self.feature_names].copy()
        y = df['loan_approved'] if 'loan_approved' in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train a Random Forest model with hyperparameter tuning."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=config.RANDOM_STATE)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        # Validate
        val_accuracy = self.model.score(X_val, y_val)
        logger.info(f"Random Forest validation accuracy: {val_accuracy:.4f}")
        
        return val_accuracy
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train an XGBoost model with hyperparameter tuning."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=config.RANDOM_STATE)
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        # Validate
        val_accuracy = self.model.score(X_val, y_val)
        logger.info(f"XGBoost validation accuracy: {val_accuracy:.4f}")
        
        return val_accuracy
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train a PyTorch neural network."""
        
        class LoanNN(nn.Module):
            def __init__(self, input_size):
                super(LoanNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Initialize model
        model = LoanNN(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training loop
        model.train()
        for epoch in range(100):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_predictions = (val_outputs > 0.5).float()
            val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
        
        self.model = model
        logger.info(f"Neural Network validation accuracy: {val_accuracy:.4f}")
        
        return val_accuracy
    
    def train(self, df=None):
        """Train the loan risk classification model."""
        if df is None:
            df = self.create_sample_data()
        
        logger.info(f"Training {self.model_type} model with {len(df)} samples")
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1-config.TRAIN_TEST_SPLIT), 
            random_state=config.RANDOM_STATE, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, 
            random_state=config.RANDOM_STATE, stratify=y_temp
        )
        
        # Train based on model type
        if self.model_type == "random_forest":
            accuracy = self.train_random_forest(X_train, y_train, X_val, y_val)
        elif self.model_type == "xgboost":
            accuracy = self.train_xgboost(X_train, y_train, X_val, y_val)
        elif self.model_type == "neural_network":
            accuracy = self.train_neural_network(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Final test evaluation
        test_accuracy = self.evaluate(X_test, y_test)
        
        self.is_trained = True
        logger.info(f"Final test accuracy: {test_accuracy:.4f}")
        
        if test_accuracy < config.TARGET_ACCURACY:
            logger.warning(f"Model accuracy {test_accuracy:.4f} is below target {config.TARGET_ACCURACY}")
        
        return {
            'validation_accuracy': accuracy,
            'test_accuracy': test_accuracy,
            'model_type': self.model_type
        }
    
    def predict_with_confidence(self, X):
        """Make predictions with confidence scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X[self.feature_names])
        else:
            X_scaled = self.scaler.transform(X)
        
        if self.model_type == "neural_network":
            # Neural network predictions
            X_tensor = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                proba = self.model(X_tensor).numpy().flatten()
            predictions = (proba > 0.5).astype(int)
            # Use prediction probability as confidence
            confidence_scores = np.where(predictions == 1, proba, 1 - proba)
        else:
            # Sklearn-based models
            predictions = self.model.predict(X_scaled)
            proba = self.model.predict_proba(X_scaled)
            # Use max probability as confidence
            confidence_scores = np.max(proba, axis=1)
        
        return predictions, confidence_scores
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        predictions, confidence_scores = self.predict_with_confidence(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        logger.info(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"Average Confidence: {np.mean(confidence_scores):.4f}")
        
        return accuracy
    
    def save_model(self, filepath):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")

def extract_loan_key_points(loan_data):
    """Extract key points from loan application data for underwriting."""
    key_points = {
        'applicant_profile': {
            'credit_score': loan_data.get('credit_score', 'Not provided'),
            'annual_income': loan_data.get('annual_income', 'Not provided'),
            'employment_length': loan_data.get('employment_length', 'Not provided'),
            'home_ownership': loan_data.get('home_ownership', 'Not provided')
        },
        'loan_details': {
            'loan_amount': loan_data.get('loan_amount', 'Not provided'),
            'loan_purpose': loan_data.get('loan_purpose', 'Not provided'),
            'debt_to_income_ratio': loan_data.get('debt_to_income_ratio', 'Not provided')
        },
        'risk_indicators': []
    }
    
    # Add risk indicators based on data
    if loan_data.get('credit_score', 0) < 600:
        key_points['risk_indicators'].append('Low credit score (below 600)')
    
    if loan_data.get('debt_to_income_ratio', 0) > 0.4:
        key_points['risk_indicators'].append('High debt-to-income ratio (>40%)')
    
    if loan_data.get('employment_length', 0) < 2:
        key_points['risk_indicators'].append('Short employment history (<2 years)')
    
    return key_points
