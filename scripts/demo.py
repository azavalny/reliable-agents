"""Demo script to showcase the Reliable Agents system capabilities."""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demo_financial_analysis():
    """Demonstrate financial loan analysis."""
    print("\n" + "="*60)
    print("🏦 FINANCIAL LOAN ANALYSIS DEMO")
    print("="*60)
    
    try:
        from src.models.financial_models import LoanRiskClassifier, extract_loan_key_points
        from src.models.model_manager import model_trainer
        import pandas as pd
        
        # Sample loan application
        sample_loan = {
            'credit_score': 720,
            'annual_income': 65000,
            'income': 65000,
            'loan_amount': 25000,
            'employment_length': 5.0,
            'debt_to_income_ratio': 0.25,
            'home_ownership': 1,  # Own
            'loan_purpose': 1     # Home Improvement
        }
        
        print("📋 Sample Loan Application:")
        for key, value in sample_loan.items():
            print(f"  {key}: {value}")
        
        print("\n🔍 Extracting Key Points...")
        key_points = extract_loan_key_points(sample_loan)
        print("✅ Key Points Extracted:")
        for category, points in key_points.items():
            print(f"  {category}: {points}")
        
        print("\n🤖 Training Financial Model...")
        model_id = model_trainer.train_financial_model('random_forest', 
                                                      model_name='Demo RF Model')
        print(f"✅ Model trained successfully! ID: {model_id}")
        
        print("\n🎯 Making Prediction...")
        from src.models.model_manager import model_manager
        model = model_manager.load_model(model_id)
        
        df = pd.DataFrame([sample_loan])
        predictions, confidence = model.predict_with_confidence(df)
        
        decision = "APPROVED" if predictions[0] == 1 else "DENIED"
        print(f"📊 RESULT: {decision}")
        print(f"🔒 Confidence: {confidence[0]:.2%}")
        
        if confidence[0] >= 0.95:
            print("✅ HIGH CONFIDENCE - Decision can proceed")
        else:
            print("⚠️ LOW CONFIDENCE - Expert review required")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def demo_medical_analysis():
    """Demonstrate medical image analysis."""
    print("\n" + "="*60)
    print("🏥 MEDICAL CHEST X-RAY ANALYSIS DEMO")
    print("="*60)
    
    try:
        from src.models.medical_models import ChestXRayClassifier, analyze_chest_xray
        from src.models.model_manager import model_trainer
        
        print("🤖 Training Medical Model...")
        model_id = model_trainer.train_medical_model('cnn', 
                                                    model_name='Demo CNN Model')
        print(f"✅ Model trained successfully! ID: {model_id}")
        
        print("\n🔍 Analyzing Sample X-rays...")
        from src.models.model_manager import model_manager
        model = model_manager.load_model(model_id)
        
        # Test with different synthetic conditions
        test_cases = [
            'synthetic_normal_0.jpg',
            'synthetic_pneumonia_0.jpg', 
            'synthetic_covid19_0.jpg',
            'synthetic_tuberculosis_0.jpg'
        ]
        
        for test_case in test_cases:
            print(f"\n📸 Analyzing: {test_case}")
            predictions, confidence = model.predict_with_confidence([test_case])
            
            diagnosis = model.classes[predictions[0]]
            print(f"🩺 Diagnosis: {diagnosis.upper()}")
            print(f"🔒 Confidence: {confidence[0]:.2%}")
            
            # Get medical analysis
            try:
                analysis = analyze_chest_xray(test_case, model)
                print(f"📋 Recommendation: {analysis['recommendation'][:100]}...")
                if analysis['requires_expert_review']:
                    print("⚠️ EXPERT REVIEW REQUIRED")
            except:
                print("📋 Basic analysis completed")
                
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def demo_llm_agent():
    """Demonstrate LLM agent capabilities."""
    print("\n" + "="*60)
    print("🤖 LLM AGENT ANALYSIS DEMO")
    print("="*60)
    
    try:
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️ No OpenAI API key found - skipping LLM demo")
            return
            
        from src.agents.llm_agent import HighPrecisionLLMAgent
        
        agent = HighPrecisionLLMAgent()
        
        print("🧠 Analyzing Task Requirements...")
        analysis = agent.analyze_task_requirements(
            "High-precision loan risk assessment for mortgage applications"
        )
        
        print("✅ Task Analysis Results:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        print("\n🔍 Interpreting Model Results...")
        interpretation = agent.interpret_results(
            predictions=[1, 0, 1], 
            confidence_scores=[0.96, 0.88, 0.99],
            task_type='financial'
        )
        
        print("✅ LLM Interpretation:")
        print(f"  {interpretation[:200]}...")
        
    except Exception as e:
        print(f"❌ LLM demo failed: {e}")

def demo_model_management():
    """Demonstrate model management capabilities."""
    print("\n" + "="*60)
    print("🔧 MODEL MANAGEMENT DEMO")  
    print("="*60)
    
    try:
        from src.models.model_manager import model_manager
        
        print("📊 Available Models:")
        financial_models = model_manager.get_available_models('financial')
        medical_models = model_manager.get_available_models('medical')
        
        print(f"  Financial Models: {len(financial_models)}")
        for model_id, info in financial_models.items():
            print(f"    - {info['name']} ({info['model_type']})")
            
        print(f"  Medical Models: {len(medical_models)}")
        for model_id, info in medical_models.items():
            print(f"    - {info['name']} ({info['model_type']})")
        
        # Show best models
        best_financial = model_manager.get_best_model('financial')
        best_medical = model_manager.get_best_model('medical')
        
        if best_financial:
            print(f"\n🏆 Best Financial Model: {best_financial}")
        if best_medical:
            print(f"🏆 Best Medical Model: {best_medical}")
            
    except Exception as e:
        print(f"❌ Model management demo failed: {e}")

def main():
    """Run all demos."""
    print("🤖 RELIABLE AGENTS SYSTEM DEMO")
    print("High-Precision AI for Financial & Medical Applications")
    print("=" * 80)
    
    # Check system requirements
    print("🔍 System Check:")
    try:
        import streamlit, openai, torch, sklearn
        print("✅ All core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return
    
    # Run demos
    demos = [
        ("Financial Analysis", demo_financial_analysis),
        ("Medical Analysis", demo_medical_analysis), 
        ("LLM Agent", demo_llm_agent),
        ("Model Management", demo_model_management)
    ]
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {demo_name} demo failed: {e}")
    
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETE!")
    print("\nTo run the full interactive system:")
    print("  python run.py")
    print("\nOr directly:")
    print("  streamlit run app.py")
    print("="*80)

if __name__ == "__main__":
    main()
