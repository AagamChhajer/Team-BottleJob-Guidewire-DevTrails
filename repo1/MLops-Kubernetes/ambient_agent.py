import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM

from typing import Dict, List
from dotenv import load_dotenv
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
load_dotenv()


# class K8sDocSearchTool(BaseModel):
#     name = "Kubernetes Documentation Search"
#     description = "Search through predefined Kubernetes documentation sources"

#     def __init__(self):
#         self.docs_sources: Dict[str, str] = {
#             "kubernetes": "https://kubernetes.io/docs/tasks/debug/debug-cluster/",
#             "lumigo": "https://lumigo.io/kubernetes-troubleshooting/",
#             "k21academy": "https://k21academy.com/docker-kubernetes/kubernetes-troubleshooting/"
#         }
#         self.cache: Dict[str, str] = {}  # Cache for storing fetched content
        
#         self.predefined_responses: Dict[str, str] = {
#             "metrics": """
#                 Common Kubernetes metrics include:
#                 - CPU utilization
#                 - Memory usage
#                 - Pod status
#                 - Node health
#                 - Network throughput
#             """,
#             "failures": """
#                 Common failure scenarios:
#                 - Resource exhaustion
#                 - Node failures
#                 - Network partitioning
#                 - Application crashes
#             """,
#             "monitoring": """
#                 Key monitoring aspects:
#                 - Resource utilization
#                 - Pod health
#                 - Service availability
#                 - Network performance
#                 - Storage metrics
#             """
#         }

#     def fetch_doc_content(self, url: str) -> str:
#         """Fetch and parse content from documentation URL"""
#         if url in self.cache:
#             return self.cache[url]
            
#         try:
#             response = requests.get(url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.text, 'html.parser')
#             # Extract main content, removing navigation and footer
#             content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
#             if content:
#                 self.cache[url] = content.get_text()
#                 return self.cache[url]
#         except Exception as e:
#             return f"Error fetching documentation: {str(e)}"
#         return ""

#     def _run(self, query: str) -> str:
#         # First check predefined responses
#         for keyword, response in self.predefined_responses.items():
#             if keyword.lower() in query.lower():
#                 return response
        
#         # Then search through documentation sources
#         for source, url in self.docs_sources.items():
#             if source.lower() in query.lower():
#                 content = self.fetch_doc_content(url)
#                 if content:
#                     return f"Found in {source} documentation:\n{content[:500]}..."  # Return first 500 chars
                
#         return "Information not found in authorized documentation sources."

# Load environment variables and model

model = joblib.load('app/Machine_Failure_classification.pkl')

# Initialize CrewAI tools
# k8s_doc_tool = K8sDocSearchTool()

# Configure the model
model_agent = LLM(
    model="openai/gpt-4o-mini"
)

# Title and Introduction
st.title("Kubernetes Cluster Health Prediction")
st.write("Predict potential machine failures in your Kubernetes cluster!")

# Streamlit User Inputs
st.sidebar.header("Cluster Metrics:")

footfall = st.sidebar.number_input("Footfall", value=0.0)
temp_mode = st.sidebar.number_input("Temperature Mode", value=0.0)
aq = st.sidebar.number_input("Air Quality (AQ)", value=0.0)
uss = st.sidebar.number_input("USS (Utilization Status)", value=0.0)
cs = st.sidebar.number_input("Cluster Status (CS)", value=0.0)
voc = st.sidebar.number_input("VOC (Volatile Organic Compounds)", value=0.0)
rp = st.sidebar.number_input("Resource Pressure (RP)", value=0.0)
ip = st.sidebar.number_input("Infrastructure Performance (IP)", value=0.0)
temperature = st.sidebar.number_input("Temperature", value=0.0)

# Create metrics dictionary
cluster_metrics = {
    "footfall": footfall,
    "tempMode": temp_mode,
    "AQ": aq,
    "USS": uss,
    "CS": cs,
    "VOC": voc,
    "RP": rp,
    "IP": ip,
    "Temperature": temperature
}

# Agent Backstories
metrics_analyzer_backstory = """
You are a Kubernetes metrics specialist with expertise in analyzing cluster performance metrics.
You can interpret various performance indicators and identify patterns that might lead to system failures.
"""

prediction_analyst_backstory = """
You are an ML model specialist focusing on prediction analysis and risk assessment.
You can evaluate prediction results and provide detailed insights about potential system failures.
"""

# Implement Agents
metrics_analyzer = Agent(
    role="Kubernetes Metrics Analyzer",
    goal="Analyze cluster metrics and identify potential issues",
    backstory=metrics_analyzer_backstory,
    verbose=True,
    allow_delegation=False,
    # tools=[k8s_doc_tool],  # Replace existing tools with k8s_doc_tool
    llm=model_agent
)

prediction_analyst = Agent(
    role="Prediction Analysis Expert",
    goal="Evaluate prediction results and assess failure risks",
    backstory=prediction_analyst_backstory,
    verbose=True,
    allow_delegation=False,
    # tools=[k8s_doc_tool],  # Replace existing tools with k8s_doc_tool
    llm=model_agent
)

# Tasks
task1 = Task(
    description=f"""
    Analyze the Kubernetes cluster metrics: {cluster_metrics}
    Identify any concerning patterns or anomalies in the metrics.
    """,
    expected_output="Detailed analysis of cluster metrics with identified patterns",
    agent=metrics_analyzer
)

task2 = Task(
    description=f"""
    Using the ML model, predict potential failures and analyze the prediction confidence.
    Current metrics: {cluster_metrics}
    """,
    expected_output="Prediction results with confidence analysis",
    agent=prediction_analyst
)

task3 = Task(
    description="""
    Generate a comprehensive report including:
    1. Current cluster health status
    2. Prediction results and confidence levels
    3. Recommended actions based on the analysis
    4. Potential risks and mitigation strategies
    """,
    expected_output="Detailed technical report with recommendations",
    agent=prediction_analyst,
    context=[task1, task2]
)

# Create Crew
crew = Crew(
    agents=[metrics_analyzer, prediction_analyst],
    tasks=[task1, task2, task3],
    verbose=True,
    planning=True
)

# Prediction Button
if st.sidebar.button("Analyze Cluster"):
    st.header("Cluster Analysis Results:")
    
    try:
        with st.spinner("Analyzing cluster health and generating predictions..."):
            # Prepare data for model prediction
            input_df = pd.DataFrame([cluster_metrics])
            model_prediction = model.predict(input_df)
            
            # Get crew analysis
            crew_output = crew.kickoff()
            
            # Display results
            st.subheader("🔍 Prediction Result")
            prediction_result = "High Risk of Failure" if model_prediction[0] == 1 else "Healthy Operation"
            st.markdown(f"### Status: {prediction_result}")
            
            # Display detailed analysis
            st.subheader("📊 Detailed Analysis")
            st.markdown(crew_output.tasks_output[0])  # Metrics analysis
            
            st.subheader("⚠️ Risk Assessment")
            st.markdown(crew_output.tasks_output[1])  # Prediction analysis
            
            st.subheader("📋 Recommendations")
            st.markdown(crew_output.tasks_output[2])  # Final report
            
            # Display analysis metrics
            st.sidebar.markdown("### 📈 Analysis Statistics")
            st.sidebar.markdown(f"""
            * Total Tokens Used: {crew_output.token_usage.total_tokens:,}
            * Analysis Confidence: {crew_output.token_usage.successful_requests * 100}%
            """)
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.info("Please verify your input values and try again.")