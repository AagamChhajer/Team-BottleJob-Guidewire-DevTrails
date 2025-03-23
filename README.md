# Kubernetes Failure Prediction System

This project aims to develop a predictive maintenance system for Kubernetes clusters using machine learning algorithms. By analyzing historical and real-time cluster metrics, the system forecasts potential failures such as node or pod crashes, resource bottlenecks, and network issues, enabling proactive management and enhanced cluster reliability.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Docker Setup](#docker-setup)
  - [Kubernetes Deployment](#kubernetes-deployment)
  - [Redis Setup](#redis-setup)
- [Running the Application](#running-the-application)
- [MLflow Integration](#mlflow-integration)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Kubernetes clusters are susceptible to various failures, including pod crashes, resource exhaustion, and network disruptions. This project integrates functionalities from existing tools to predict such issues before they occur by analyzing key metrics like CPU usage, memory consumption, pod status, and network I/O. The system employs machine learning techniques, such as anomaly detection and time-series analysis, to forecast potential failures, thereby improving the resilience and reliability of Kubernetes clusters.

## Technologies Used

- **Streamlit**: Interactive UI for user input and results visualization.
- **Redis**: In-memory data structure store used to cache predictions.
- **Docker**: Containerization of the application for consistent deployment.
- **Kubernetes**: Management of deployment, scaling, and orchestration of Docker containers.
- **MLflow**: Tracking of model performance and metrics.
- **Scikit-learn**: Machine learning algorithms for classification tasks.
- **Pandas**: Data manipulation and analysis.
- **Python 3.10**: Programming language used for development.

## Setup and Installation

### Prerequisites

- **Docker**: Install Docker from [here](https://docs.docker.com/get-docker/).
- **Kubernetes**: Install Kubernetes (minikube recommended) from [here](https://kubernetes.io/docs/tasks/tools/).
- **Redis**: Install Redis from [here](https://redis.io/download) or run it using Docker.
- **Python 3.10**: Ensure Python is installed on your system.

### Clone the Repository

```bash
git clone https://github.com/yourusername/k8s-failure-prediction.git
cd k8s-failure-prediction
```

### Install Dependencies

Install the required Python packages listed in the `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Docker Setup

1. **Build the Docker image:**

   ```bash
   docker build -t k8s-failure-prediction-app .
   ```

2. **Run the Docker container:**

   ```bash
   docker run -p 8501:8501 k8s-failure-prediction-app
   ```

### Kubernetes Deployment

1. **Apply the Kubernetes deployment and service files:**

   ```bash
   kubectl apply -f kubernetes-deployment.yaml
   kubectl apply -f kubernetes-service.yaml
   ```

2. **Access the app by finding the external IP of the service:**

   ```bash
   kubectl get services
   ```

### Redis Setup

1. **Start the Redis server locally:**

   ```bash
   redis-server
   ```

2. **Or run Redis in a Docker container:**

   ```bash
   docker run --name redis -p 6379:6379 -d redis
   ```

## Running the Application

To start the Streamlit application locally, run:

```bash
streamlit run app.py
```

The app allows users to input features such as `CPU usage`, `memory usage`, `pod status`, and `network I/O`, and get predictions on potential failures in the Kubernetes cluster.

## MLflow Integration

The project logs the trained model and metrics (e.g., accuracy) using MLflow:

```python
import mlflow.sklearn

with mlflow.start_run():
    mlflow.sklearn.log_model(model, 'k8s_failure_prediction_model.pkl')
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_artifact('cluster_metrics.csv')
```

To run the MLflow script:

```bash
python mlflow_integration.py
```

## Contributing

Contributions are welcome! Feel free to submit a pull request or raise an issue.

## License

This project is licensed under the MIT License.
