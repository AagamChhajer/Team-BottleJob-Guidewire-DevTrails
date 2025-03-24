# Kubernetes Cluster Health Prediction System with CrewAI Agents

## Overview
This project implements an intelligent system for predicting and analyzing Kubernetes cluster health using CrewAI agents and machine learning. The system combines real-time metric analysis with ML predictions to provide comprehensive cluster health assessments and automated troubleshooting recommendations.

## Core Components

### 1. Machine Learning Model
- **Random Forest Classifier** for failure prediction
- **Preprocessing Pipeline** with StandardScaler
- **Model Metrics**: Accuracy-based evaluation
- **Model Persistence**: Saved as 'Machine_Failure_classification.pkl'

### 2. CrewAI Agent Implementation

#### Agents Structure
1. **Metrics Analyzer Agent**
   ```python
   metrics_analyzer = Agent(
       role="Kubernetes Metrics Analyzer",
       goal="Analyze cluster metrics and identify potential issues",
       backstory="Kubernetes metrics specialist with expertise in analyzing cluster performance metrics",
       tools=[FirecrawlScrapeWebsiteTool],
       llm=model_agent
   )
   ```

2. **Prediction Analyst Agent**
   ```python
   prediction_analyst = Agent(
       role="Prediction Analysis Expert",
       goal="Evaluate prediction results and assess failure risks",
       backstory="ML model specialist focusing on prediction analysis and risk assessment",
       tools=[FirecrawlScrapeWebsiteTool],
       llm=model_agent
   )
   ```

#### Task Pipeline
1. **Metric Analysis Task**
   - Analyzes cluster metrics for patterns
   - Identifies anomalies in performance

2. **Prediction Task**
   - Utilizes ML model for failure prediction
   - Provides confidence scores

3. **Report Generation Task**
   - Compiles comprehensive analysis
   - Generates actionable recommendations

### 3. Monitored Metrics
- Footfall (Resource Usage)
- Temperature Mode
- Air Quality (AQ)
- Utilization Status (USS)
- Cluster Status (CS)
- Volatile Organic Compounds (VOC)
- Resource Pressure (RP)
- Infrastructure Performance (IP)
- Temperature

## Technical Implementation

### Data Processing Pipeline
```python
preprocess_pipeline = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)

model_pipeline = Pipeline(
    steps=[
        ('model', RandomForestClassifier())
    ]
)
```

### Agent Tools Integration
```python
tool = FirecrawlScrapeWebsiteTool(
    url='https://komodor.com/learn/kubernetes-troubleshooting-the-complete-guide/',
    page_options='includeHtml'
)
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip
- virtualenv
- OpenAI API key

### Installation Steps
1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Team-BottleJob-Guidewire-DevTrails
   ```

2. **Environment Setup**
   ```bash
   python -m venv ven
   .\ven\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   # Create .env file
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application
```bash
streamlit run .\repo1\MLops-Kubernetes\ambient_agent.py
```

## Agent Workflow

1. **Data Collection**
   - User inputs cluster metrics via Streamlit interface
   - System prepares data for analysis

2. **Agent Processing**
   ```python
   crew = Crew(
       agents=[metrics_analyzer, prediction_analyst],
       tasks=[task1, task2, task3],
       verbose=True,
       planning=True
   )
   ```

3. **Analysis Generation**
   - Metrics analysis by Metrics Analyzer
   - Failure prediction by ML model
   - Risk assessment by Prediction Analyst
   - Comprehensive report compilation

## Output Components

### 1. Prediction Results
- Binary health status (Healthy/Risk of Failure)
- Confidence metrics
- Token usage statistics

### 2. Analysis Report
- Detailed metric patterns
- Anomaly detection results
- Performance indicators
- Risk assessment

### 3. Recommendations
- Preventive measures
- Optimization suggestions
- Troubleshooting steps

## Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License
[Your chosen license]

## Authors
Team BottleJob

## Acknowledgments
- CrewAI framework
- Kubernetes documentation
- OpenAI
- Scikit-learn community

## Project Structure
```
Team-BottleJob-Guidewire-DevTrails/
├── app/
│   ├── model.py                    # ML model implementation
│   └── Machine_Failure_classification.pkl
├── repo1/MLops-Kubernetes/
│   └── ambient_agent.py           # CrewAI agent implementation
├── scripts/
│   └── preprocess.py              # Data preprocessing
├── requirements.txt
└── README.md
```
