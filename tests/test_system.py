"""Test script to validate the Reliable Agents system."""

import logging
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_financial_models():
    """Test financial model training and prediction."""
    logger.info("Testing financial models...")
    
    try:
        from src.models.financial_models import LoanRiskClassifier
        from src.models.model_manager import model_manager, model_trainer
        
        # Test Random Forest
        logger.info("Training Random Forest model...")
        model_id = model_trainer.train_financial_model('random_forest', 
                                                      model_name='Test RF Model')
        
        # Load and test prediction
        model = model_manager.load_model(model_id)
        
        # Test with sample data
        test_data = pd.DataFrame([{
            'credit_score': 720,
            'annual_income': 65000,
            'income': 65000,
            'loan_amount': 25000,
            'employment_length': 5.0,
            'debt_to_income_ratio': 0.25,
            'home_ownership': 1,
            'loan_purpose': 1
        }])
        
        predictions, confidence = model.predict_with_confidence(test_data)
        logger.info(f"Financial prediction: {predictions[0]}, confidence: {confidence[0]:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Financial model test failed: {e}")
        return False

def test_medical_models():
    """Test medical model training and prediction."""
    logger.info("Testing medical models...")
    
    try:
        from src.models.medical_models import ChestXRayClassifier
        from src.models.model_manager import model_manager, model_trainer
        
        # Test CNN model (smaller for testing)
        logger.info("Training CNN model...")
        model_id = model_trainer.train_medical_model('cnn', 
                                                    model_name='Test CNN Model')
        
        # Load and test prediction
        model = model_manager.load_model(model_id)
        
        # Test with synthetic data
        predictions, confidence = model.predict_with_confidence(['synthetic_normal_0.jpg'])
        logger.info(f"Medical prediction: {predictions[0]}, confidence: {confidence[0]:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Medical model test failed: {e}")
        return False

def test_model_management():
    """Test model management functionality."""
    logger.info("Testing model management...")
    
    try:
        from src.models.model_manager import model_manager
        
        # Get available models
        financial_models = model_manager.get_available_models('financial')
        medical_models = model_manager.get_available_models('medical')
        
        logger.info(f"Found {len(financial_models)} financial models")
        logger.info(f"Found {len(medical_models)} medical models")
        
        # Test best model selection
        best_financial = model_manager.get_best_model('financial')
        best_medical = model_manager.get_best_model('medical')
        
        if best_financial:
            logger.info(f"Best financial model: {best_financial}")
        if best_medical:
            logger.info(f"Best medical model: {best_medical}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model management test failed: {e}")
        return False

def test_llm_agent():
    """Test LLM agent functionality (if API key available)."""
    logger.info("Testing LLM agent...")
    
    try:
        from src.agents.llm_agent import HighPrecisionLLMAgent
        
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning("No OpenAI API key found. Skipping LLM agent test.")
            return True
        
        agent = HighPrecisionLLMAgent()
        
        # Test task analysis
        analysis = agent.analyze_task_requirements("Loan risk assessment for mortgage application")
        logger.info(f"Task analysis successful: {analysis.get('task_type', 'Unknown')}")
        
        # Test result interpretation
        interpretation = agent.interpret_results([1], [0.96], 'financial')
        logger.info(f"Result interpretation generated: {len(interpretation)} characters")
        
        return True
        
    except Exception as e:
        logger.error(f"LLM agent test failed: {e}")
        return False

def run_all_tests():
    """Run all system tests."""
    logger.info("Starting Reliable Agents system tests...")
    
    tests = [
        ("Financial Models", test_financial_models),
        ("Medical Models", test_medical_models),
        ("Model Management", test_model_management),
        ("LLM Agent", test_llm_agent)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: FAILED - {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
