"""Main LLM Agent for high-precision financial and medical applications."""

import json
import logging
from typing import Dict, List, Tuple, Any
from openai import OpenAI
from pathlib import Path
from ..utils import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighPrecisionLLMAgent:
    """Main LLM agent for coordinating specialized model training and inference."""
    
    def __init__(self):
        """Initialize the LLM agent with OpenAI client."""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.available_models = self._discover_available_models()
        
    def _discover_available_models(self) -> Dict[str, List[str]]:
        """Discover available pre-trained models."""
        models = {"financial": [], "medical": []}
        
        # Check for existing models in the models directory
        if config.MODELS_DIR.exists():
            for model_file in config.MODELS_DIR.glob("*.joblib"):
                if "financial" in model_file.stem:
                    models["financial"].append(model_file.stem)
                elif "medical" in model_file.stem:
                    models["medical"].append(model_file.stem)
                    
        return models
    
    def generate_model_code(self, task_type: str, model_type: str, features: List[str]) -> str:
        """Generate Python code for training a specialized model."""
        
        system_prompt = f"""You are an expert machine learning engineer specializing in high-precision models for {task_type} applications.
        Generate Python code to train a {model_type} model with 99%+ accuracy requirements.
        
        Requirements:
        - Use appropriate validation techniques
        - Include confidence scoring
        - Implement proper error handling
        - Target accuracy: {config.TARGET_ACCURACY}
        - Minimum confidence threshold: {config.MIN_CONFIDENCE_THRESHOLD}
        - Save the trained model to disk
        """
        
        user_prompt = f"""Generate complete Python code to train a {model_type} model for {task_type} task.
        
        Features to use: {features}
        Model types available: {config.FINANCIAL_MODEL_TYPES if task_type == 'financial' else config.MEDICAL_MODEL_TYPES}
        
        The code should:
        1. Load and preprocess data
        2. Split data properly with validation
        3. Train the model with hyperparameter tuning
        4. Evaluate with multiple metrics
        5. Save the model with metadata
        6. Return predictions with confidence scores
        
        Make it production-ready with proper error handling."""
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating model code: {e}")
            raise
    
    def analyze_task_requirements(self, task_description: str) -> Dict[str, Any]:
        """Analyze task requirements and suggest optimal approach."""
        
        system_prompt = """You are an expert in high-precision ML applications for finance and healthcare.
        Analyze the task and provide recommendations in JSON format."""
        
        user_prompt = f"""Analyze this task: {task_description}
        
        Provide recommendations in this JSON format:
        {{
            "task_type": "financial" or "medical",
            "recommended_model": "best model type",
            "confidence_requirements": "explanation",
            "data_requirements": ["list", "of", "requirements"],
            "preprocessing_steps": ["step1", "step2"],
            "validation_strategy": "description"
        }}"""
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content
            # Find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Error analyzing task requirements: {e}")
            # Return default analysis
            return {
                "task_type": "financial",
                "recommended_model": "random_forest",
                "confidence_requirements": "99% accuracy required",
                "data_requirements": ["clean_data", "balanced_classes"],
                "preprocessing_steps": ["normalize", "encode_categorical"],
                "validation_strategy": "stratified_k_fold"
            }
    
    def interpret_results(self, predictions: List[float], confidence_scores: List[float], 
                         task_type: str) -> str:
        """Interpret model results and provide human-readable explanation."""
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        high_confidence_count = sum(1 for c in confidence_scores if c >= config.MIN_CONFIDENCE_THRESHOLD)
        
        system_prompt = f"""You are an expert {task_type} analyst. Interpret ML model results 
        and provide clear, actionable insights for decision makers."""
        
        user_prompt = f"""Interpret these model results:
        
        Predictions: {predictions[:5]}...  (showing first 5)
        Average Confidence: {avg_confidence:.3f}
        High Confidence Predictions: {high_confidence_count}/{len(predictions)}
        Minimum Required Confidence: {config.MIN_CONFIDENCE_THRESHOLD}
        
        Task Type: {task_type}
        
        Provide:
        1. Summary of results quality
        2. Confidence assessment
        3. Recommendations for action
        4. Any risk factors to consider"""
        
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error interpreting results: {e}")
            return f"Model completed with {avg_confidence:.1%} average confidence. {high_confidence_count}/{len(predictions)} predictions meet the {config.MIN_CONFIDENCE_THRESHOLD:.1%} threshold."
