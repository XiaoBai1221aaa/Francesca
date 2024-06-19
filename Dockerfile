name: Deploy to Cloud Run

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout the code
      uses: actions/checkout@v2

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v2.1.0
      with:
        project_id: ${{ secrets.GCP_PROJECT_ID }}
        service_account_key: ${{ secrets.GCP_SA_KEY }}

    - name: Configure Docker with gcloud
      run: gcloud auth configure-docker

    - name: Build the Docker image
      run: docker build -t gcr.io/${{ secrets.GCP_PROJECT_ID }}/francesca-app:$GITHUB_SHA .

    - name: Push the Docker image
      run: docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/francesca-app:$GITHUB_SHA

    - name: Deploy to Cloud Run
      run: gcloud run deploy francesca-app --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/francesca-app:$GITHUB_SHA --platform managed --region us-central1 --quiet
