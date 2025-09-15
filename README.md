# Reliable Agents - High Precision AI System

An advanced LLM agent system designed for high-precision financial and medical applications requiring 99%+ accuracy. The system uses specialized machine learning models trained for specific tasks and provides confidence scoring for all predictions.

## Features

### Financial Applications
- **Loan Underwriting**: Extract key points from loan documents and assess creditworthiness
- **Risk Assessment**: Real-time loan approval decisions with confidence scoring
- **Compliance**: Meets regulatory requirements for financial decision-making

### Medical Applications  
- **Chest X-ray Analysis**: Detect pneumonia, COVID-19, tuberculosis, and normal cases
- **High-Precision Diagnostics**: 99%+ accuracy with expert review recommendations
- **Confidence Scoring**: Each prediction includes confidence levels for clinical decision support

## Architecture

The system consists of several key components:

1. **Main LLM Agent** (`llm_agent.py`): Orchestrates the entire system using OpenAI's GPT models
2. **Specialized Models**: 
   - Financial models (`financial_models.py`): Random Forest, XGBoost, Neural Networks
   - Medical models (`medical_models.py`): CNN, ResNet, traditional ML models
3. **Model Manager** (`model_manager.py`): Handles model storage, versioning, and selection
4. **Streamlit UI** (`app.py`): User-friendly interface for all operations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Windows/Linux/macOS

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd reliable-agents
pip install -r requirements.txt
```

2. **Configure API key**:
Create a `.env` file (copy from `env_example.txt`):
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Run the application**:
```bash
python main.py
```

Or alternatively:
```bash
streamlit run ui/app.py
```

4. **Open your browser** to `http://localhost:8501`

## 📊 Usage

### Financial Analysis
1. Navigate to "Financial Analysis" in the sidebar
2. Choose input method:
   - **Manual Entry**: Enter loan details directly
   - **Upload CSV**: Use your own loan data
   - **Sample Data**: Try pre-configured examples
3. Select a trained model or train a new one
4. Get instant loan approval decisions with confidence scores

### Medical Diagnosis
1. Navigate to "Medical Diagnosis" in the sidebar  
2. Upload a chest X-ray image or use sample images
3. Select a medical model (CNN, ResNet, etc.)
4. Get disease classification with confidence scores and treatment recommendations

### Model Management
- **View Models**: See all trained models and their performance metrics
- **Train New Models**: Create custom models with your data
- **Model Comparison**: Compare different model types and accuracies

## 🔧 Model Types

### Financial Models
- **Random Forest**: Fast training, interpretable results
- **XGBoost**: High accuracy, handles missing data well  
- **Neural Network**: Deep learning approach, best for complex patterns

### Medical Models
- **CNN**: Custom convolutional neural network
- **ResNet**: Transfer learning with pre-trained ResNet-50
- **XGBoost**: Traditional ML with extracted image features
- **Random Forest**: Fast inference with statistical image features

## 📈 Performance Standards

- **Target Accuracy**: 99%+
- **Minimum Confidence**: 95%
- **Expert Review**: Triggered for predictions below confidence threshold
- **Real-time**: Sub-second inference times

## 🛡️ Safety & Compliance

### Financial
- Regulatory compliance for loan decisions
- Audit trails for all predictions
- Bias detection and mitigation
- Risk level categorization

### Medical  
- Expert radiologist review for low-confidence cases
- Treatment recommendations based on diagnosis
- Urgency level classification
- Clinical decision support integration

## 📁 Project Structure

```
reliable-agents/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── src/                       # Source code
│   ├── agents/               # LLM agents
│   │   └── llm_agent.py     # Main LLM orchestration
│   ├── models/              # ML models
│   │   ├── financial_models.py  # Loan underwriting
│   │   ├── medical_models.py    # Medical classification
│   │   └── model_manager.py     # Model management
│   └── utils/               # Utilities
│       └── config.py        # Configuration
├── ui/                       # User interface
│   └── app.py               # Streamlit web app
├── scripts/                 # Utility scripts
│   ├── run.py              # Startup script
│   ├── demo.py             # Demonstration
│   ├── install.bat         # Windows installer
│   └── install.sh          # Unix installer
├── tests/                   # Test files
│   └── test_system.py      # System tests
├── data/                    # Data storage
│   └── samples/            # Sample datasets
├── models/                  # Trained models
└── README.md               # Documentation
```

## 🔄 Training Custom Models

### Financial Models
```python
from src.models.model_manager import model_trainer

# Train with default synthetic data
model_id = model_trainer.train_financial_model('random_forest')

# Train with custom data
model_id = model_trainer.train_financial_model('xgboost', your_data)
```

### Medical Models  
```python
# Train with synthetic X-ray data
model_id = model_trainer.train_medical_model('cnn')

# Train with custom medical images
model_id = model_trainer.train_medical_model('resnet', image_paths, labels)
```

## 🔍 API Usage

### Financial Prediction
```python
from src.models.financial_models import LoanRiskClassifier
from src.models.model_manager import model_manager

# Load trained model
model = model_manager.load_model('financial_model_id')

# Make prediction
predictions, confidence = model.predict_with_confidence(loan_data)
```

### Medical Prediction
```python
from src.models.medical_models import ChestXRayClassifier
from src.models.model_manager import model_manager

# Load trained model  
model = model_manager.load_model('medical_model_id')

# Analyze X-ray
predictions, confidence = model.predict_with_confidence(['xray.jpg'])
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes

### Financial Use
- This system is for demonstration purposes
- Ensure compliance with local financial regulations  
- Validate model performance with real-world data
- Implement proper audit trails for production use

### Medical Use
- Not intended for direct clinical diagnosis
- Always require expert radiologist review
- Validate with clinical datasets before deployment
- Ensure HIPAA compliance for patient data

## 🆘 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the example code in `/data/samples/`

## 🚦 Status

- ✅ Core LLM agent implementation
- ✅ Financial loan underwriting models  
- ✅ Medical chest X-ray classification
- ✅ Model management system
- ✅ Streamlit web interface
- ✅ Sample data and examples
- 🔄 Advanced model interpretability (coming soon)
- 🔄 Production deployment guides (coming soon)
