# Simple AI Training Job

This project demonstrates how to train a simple scikit-learn model and deploy it as a custom training job on Google AI Platform.

## Prerequisites

1.  **Google Cloud SDK:** Make sure you have the `gcloud` command-line tool installed and configured.
2.  **Docker:** Docker must be installed and running on your local machine.
3.  **Enable APIs:** Enable the AI Platform Training and Prediction API for your GCP project.

## How to Launch the Training Job

1.  **Set your GCP Project ID:**

    ```bash
    export PROJECT_ID="your-gcp-project-id"
    gcloud config set project $PROJECT_ID
    ```

2.  **Build the Docker image:**

    ```bash
    docker build -t gcr.io/$PROJECT_ID/simple-ai-training:latest .
    ```

3.  **Push the Docker image to Google Container Registry (GCR):**

    ```bash
    docker push gcr.io/$PROJECT_ID/simple-ai-training:latest
    ```

4.  **Submit the training job:**

    ```bash
    gcloud ai custom-jobs create --region=us-central1 --display-name=simple-ai-training-job --config=custom_training_job.yaml
    ```

5.  **Monitor the job:**

    You can monitor the job's progress in the Google Cloud Console under AI Platform > Jobs.