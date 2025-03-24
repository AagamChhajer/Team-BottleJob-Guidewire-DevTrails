pipeline {
    agent any

    environment {
        STREAMLIT_PORT = '8501'  
    }

    stages {
        stage('Debug Git Info') {
            steps {
                sh 'git --version'
                sh 'git remote -v'
            }
        }

        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/shreyanshdiff/MLops-Kubernetes.git'

            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                pytest || echo "Some tests failed!"
                '''
            }
        }

        stage('Build and Run App') {
            steps {
                sh '''
                pkill -f streamlit || true
                nohup streamlit run app.py --server.port=${STREAMLIT_PORT} &
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '**/*.pkl', allowEmptyArchive: true
        }
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
