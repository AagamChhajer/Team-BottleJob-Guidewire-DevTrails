Here is the `README.md` file for your project titled **"Machine Failure Classification using MLOps"**:

---

# Machine Failure Classification using MLOps

This project demonstrates the deployment of a machine learning model to predict machine failures, integrating various MLOps tools such as Streamlit for a user-friendly interface, Redis for caching, Docker for containerization, and Kubernetes for orchestration. MLflow is used for tracking model performance and metrics, ensuring a full end-to-end pipeline in production.

Link to the Application : https://mlops-kubernetes-c3obagdkuruscnpqgvgwj8.streamlit.app/

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
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Project Structure](#project-structure)
- [MLflow Integration](#mlflow-integration)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is designed to classify machine failures using a trained machine learning model. The application provides a scalable MLOps pipeline using Docker, Kubernetes, Redis, and MLflow. The key components are:
- **Streamlit App**: A web-based interface for interacting with the model.
- **Redis**: In-memory store used for caching model predictions to optimize performance.
- **Docker**: Containerizes the application for easy deployment.
- **Kubernetes**: Manages the deployment at scale, ensuring reliability.
- **MLflow**: Tracks the model, logs performance metrics, and stores artifacts.

## Technologies Used

- **Streamlit**: Interactive UI for user input and results visualization.
- **Redis**: In-memory data structure store used to cache predictions.
- **Docker**: Used to containerize the application.
- **Kubernetes**: For managing deployment, scaling, and orchestration of the Docker container.
- **MLflow**: For model logging and tracking.
- **Scikit-learn**: For the machine learning classification algorithm.
- **Pandas**: For data manipulation.
- **Joblib**: For saving and loading machine learning models.
- **Python 3.10**: The language used for development.

## Setup and Installation

### Prerequisites

- **Docker**: Install Docker from [here](https://www.docker.com/get-started).
- **Kubernetes**: Install Kubernetes (minikube recommended) from [here](https://minikube.sigs.k8s.io/docs/start/).
- **Redis**: Install Redis from [here](https://redis.io/download) or run it using Docker.
- **Python 3.10**: Make sure Python is installed on your system.

### Clone the Repository

```bash
git clone https://github.com/shreyanshdiff/machine-failure-classification
cd machine-failure-classification
```

### Install Dependencies

Install the required Python packages listed in the `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t machine-failure-app .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 machine-failure-app
   ```

### Kubernetes Deployment

1. Apply the Kubernetes deployment and service files:
   ```bash
   kubectl apply -f kubernetes-deployment.yaml
   kubectl apply -f kubernetes-service.yaml
   ```

2. Access the app by finding the external IP of the service:
   ```bash
   kubectl get services
   ```

### Redis Setup

1. Start the Redis server locally:
   ```bash
   redis-server
   ```

2. Or run Redis in a Docker container:
   ```bash
   docker run --name redis -p 6379:6379 -d redis
   ```

## Running the Streamlit Application

To start the Streamlit application locally, simply run:

```bash
streamlit run app.py
```

The app allows users to input features such as `footfall`, `temperature`, and `air quality`, and get predictions on whether the machine will fail or not.

### Streamlit UI

- Input fields for the machine features.
- Prediction result (either **Failure** or **No Failure**).


## MLflow Integration

The project logs the trained model and metrics (e.g., accuracy) using MLflow:

```python
import mlflow.sklearn

with mlflow.start_run():
    mlflow.sklearn.log_model(model, 'Failure_classification.pkl')
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_artifact('machine_failure.csv')
```

To run the MLflow script:
```bash
python mlflow_integration.py
```

## Contributing

Contributions are welcome! Feel free to submit a pull request or raise an issue.

## License

This project is licensed under the MIT License.

---

This `README.md` should provide all the information needed to get started with your project and effectively run the machine failure classification app using MLOps tools.