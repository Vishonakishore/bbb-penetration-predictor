# Deployment

The app is ready for a Python host or a Docker-based host. Runtime model files and report images must ship with the source because inference does not retrain the model and the performance page reads its saved charts.

## Docker

Build and run locally to verify the production image:

```powershell
docker build -t bbbp-predictor .
docker run --rm -p 8501:8501 bbbp-predictor
```

Open `http://localhost:8501`. The image exposes the Streamlit health endpoint at `/_stcore/health` and uses the host-provided `PORT` when set.

## Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Confirm `models/random_forest.joblib`, `models/metadata.json`, and the PNGs in `reports/` are included. They are needed for predictions and performance charts.
3. In Streamlit Community Cloud, create an app from that repository, choose the project branch, and set the main file to `app.py`.

No secrets or API keys are required. `data/BBBP.csv` is not required at runtime because it is used only to retrain the model. To update the deployed model, run `python train_model.py` locally and include the regenerated model, metadata, and report charts in the deployment source.

## Deployment state

This workspace has no Git remote, hosting account connection, or Docker executable. The deployment files are prepared, but a public service cannot be created until the source repository and target hosting account are connected.
