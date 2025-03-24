# Kubernetes Cluster Health Prediction System

## Overview
This project implements an intelligent system for predicting and analyzing Kubernetes cluster health using CrewAI agents and machine learning. The system combines real-time metric analysis with ML predictions to provide comprehensive cluster health assessments.

## Features
- 🔍 Real-time cluster metric monitoring
- 🤖 AI-powered health analysis
- 📊 ML-based failure prediction
- 📋 Automated report generation
- 🛠️ Troubleshooting recommendations

## System Architecture

### Components
1. **User Interface (Streamlit)**
   - Interactive metric input dashboard
   - Real-time analysis display
   - Visual health indicators

2. **AI Agents**
   - Metrics Analyzer: Evaluates cluster performance metrics
   - Prediction Analyst: Handles ML predictions and risk assessment

3. **Tools & Integrations**
   - Kubernetes Documentation Tool
   - Machine Learning Model
   - FirecrawlScrapeWebsiteTool for documentation access

## Monitored Metrics
- Footfall (Resource Usage)
- Temperature Mode
- Air Quality (AQ)
- Utilization Status (USS)
- Cluster Status (CS)
- Volatile Organic Compounds (VOC)
- Resource Pressure (RP)
- Infrastructure Performance (IP)
- Temperature

## Setup Instructions

### Prerequisites
```bash
python 3.8+
pip
virtualenv
```

### Installation
1. Clone the repository
```bash
git clone <repository-url>
cd Team-BottleJob-Guidewire-DevTrails
```

2. Create and activate virtual environment
```bash
python -m venv ven
.\ven\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
# Create .env file
OPENAI_API_KEY=your_api_key_here
```

### Running the Application
```bash
streamlit run .\repo1\MLops-Kubernetes\ambient_agent.py
```

## How It Works

1. **Data Collection**
   - User inputs cluster metrics through the Streamlit interface
   - System collects real-time performance data

2. **Analysis Process**
   - Metrics Analyzer evaluates current cluster state
   - ML model predicts potential failures
   - Prediction Analyst provides detailed risk assessment

3. **Output Generation**
   - Health status determination
   - Detailed metric analysis
   - Risk assessment report
   - Actionable recommendations

## Output Components

1. **Prediction Result**
   - Binary health status (Healthy/Risk of Failure)
   - Confidence score

2. **Detailed Analysis**
   - Metric patterns
   - Anomaly detection
   - Performance indicators

3. **Risk Assessment**
   - Threat levels
   - Vulnerability analysis
   - Impact assessment

4. **Recommendations**
   - Preventive measures
   - Optimization suggestions
   - Troubleshooting steps

## Usage Example
1. Input cluster metrics in the sidebar
2. Click "Analyze Cluster"
3. Review the comprehensive analysis:
   - Overall health status
   - Detailed metric analysis
   - Risk assessment
   - Recommended actions

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