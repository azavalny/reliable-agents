"""Configuration settings for the Reliable Agents system."""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to main project directory
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SAMPLE_DATA_DIR.mkdir(exist_ok=True)

# OpenAI API settings
OPENAI_MODEL = "gpt-4o-mini"  # Using available model instead of gpt-5-nano-2025-08-07
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model settings
MIN_CONFIDENCE_THRESHOLD = 0.95
TARGET_ACCURACY = 0.99

# Financial model settings
FINANCIAL_MODEL_TYPES = ["random_forest", "xgboost", "neural_network"]
FINANCIAL_FEATURES = [
    "credit_score", "income", "debt_to_income_ratio", "loan_amount", 
    "employment_length", "home_ownership", "loan_purpose", "annual_income"
]

# Medical model settings
MEDICAL_MODEL_TYPES = ["cnn", "resnet", "xgboost", "random_forest"]
MEDICAL_CLASSES = ["pneumonia", "normal", "covid19", "tuberculosis"]

# Training parameters
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42
