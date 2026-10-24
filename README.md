# Rain Prediction App

An interactive Streamlit app that estimates the probability of rain tomorrow from today's weather conditions. Explore how the forecast changes as you adjust the inputs, inspect the model's feature contributions, and compare a what-if scenario.

## Features

- Interactive controls for temperature, humidity, pressure, wind speed, cloud cover, and whether it rained today.
- Rain probability, dry probability, and an easy-to-read forecast.
- Three quick weather presets and selectable visual themes.
- Animated weather effects that can be turned off.
- Charts showing feature contributions and how changing one condition affects the forecast.
- Model performance metrics, including accuracy, precision, recall, ROC AUC, and the confusion matrix.

## Run locally

Requires Python and pip.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL in the terminal after startup. The app loads `rain_model.pkl` when available and can build the model from `rain.csv` if needed; keep `model_utils.py` alongside `app.py`.

## Model and data

The app uses a logistic regression pipeline with standardized inputs. It predicts `RainTomorrow` from these six features:

| Feature | Meaning |
| --- | --- |
| `Temperature` | Today's temperature in degrees Celsius |
| `Humidity` | Today's humidity (%) |
| `Pressure` | Today's pressure (hPa) |
| `WindSpeed` | Today's wind speed (km/h) |
| `CloudCover` | Today's cloud cover (%) |
| `RainToday` | Whether it rained today (0 = no, 1 = yes) |

The training helper removes duplicate rows, evaluates the model using a stratified 80/20 train/test split and five-fold cross-validation on the training split, then refits the deployment model on all available rows. The displayed feature contributions are based on the fitted logistic regression coefficients and standardized feature values; they are explanatory aids, not causal effects.

The dataset is provided as `rain.csv`. Check its source, license, and terms before reusing or redistributing it.

To retrain the model bundle after changing the dataset:

```bash
python train.py
```

This writes `rain_model.pkl`, which the app loads at startup.

## Repository files

```text
app.py             Streamlit interface, forecast visualization, and interactions
model_utils.py     Model training, evaluation, and bundle creation
train.py           Script to retrain and save the model bundle
rain.csv           Weather observations used for training
rain_model.pkl     Serialized model bundle used by the app
rain.pkl           Additional serialized model artifact
model.ipynb        Model exploration notebook
requirements.txt   Python dependencies
```

## Limitations

This app is an educational demonstration, not an operational weather forecasting service. Forecast quality depends on the training data and may not generalize to other locations, seasons, or real-time conditions. Treat the displayed probability and explanations as estimates, not as authoritative forecasts.

## Built with

Python, Streamlit, pandas, NumPy, scikit-learn, joblib, and Plotly.
