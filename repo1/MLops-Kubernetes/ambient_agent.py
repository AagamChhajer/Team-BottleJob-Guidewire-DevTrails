import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from typing import Dict, List, ClassVar, Optional
from dotenv import load_dotenv
import joblib
import pandas as pd
from crewai_tools import FirecrawlScrapeWebsiteTool
load_dotenv()


tool = FirecrawlScrapeWebsiteTool(url='https://komodor.com/learn/kubernetes-troubleshooting-the-complete-guide/',page_options = 'includeHtml')



# Load environment variables and model

model = joblib.load("C:\\Users\\HP\Desktop\\Team-BottleJob-Guidewire-DevTrails\\repo1\\MLops-Kubernetes\\app\\Machine_Failure_classification.pkl")

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
You can evaluate prediction results and provide insights and troubleshoots about potential system failures.
"""

# Implement Agents
metrics_analyzer = Agent(
    role="Kubernetes Metrics Analyzer",
    goal="Analyze cluster metrics and identify potential issues",
    backstory=metrics_analyzer_backstory,
    verbose=True,
    allow_delegation=False,
    tools=[tool],  # Replace existing tools with k8s_doc_tool
    llm=model_agent
)

prediction_analyst = Agent(
    role="Prediction Analysis Expert",
    goal="Evaluate prediction results and assess failure risks and provide troubleshoots",
    backstory=prediction_analyst_backstory,
    verbose=True,
    allow_delegation=False,
    tools=[tool],  # Replace existing tools with k8s_doc_tool
    llm=model_agent
)

# Tasks
task1 = Task(
    description=f"""
    Analyze the Kubernetes cluster metrics: {cluster_metrics}
    Identify any concerning patterns or anomalies in the metrics.
    """,
    expected_output="Concise analysis of cluster metrics with identified patterns",
    agent=metrics_analyzer
)

task2 = Task(
    description=f"""
    Using the ML model, predict potential failures.
    Current metrics: {cluster_metrics}
    """,
    expected_output="Prediction results with smart and intelligent analysis",
    agent=prediction_analyst
)

task3 = Task(
    description="""
    Generate a comprehensive report including:
    1. Current cluster health status
    2. Prediction results
    3. Recommended actions based on the analysis
    4. Potential risks and mitigation strategies
    """,
    expected_output="Directed and Concise Technical report with recommendations",
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