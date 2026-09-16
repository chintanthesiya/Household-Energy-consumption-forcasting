# Household Energy Consumption Forecasting

An interactive Streamlit dashboard for exploring household electricity usage and estimating daily energy consumption from household, weather, air-conditioning, peak-usage, and calendar inputs.

The project combines a trained scikit-learn/XGBoost-compatible model with a practical dashboard so users can inspect the historical dataset, understand consumption patterns, enter a household profile, and receive an immediate prediction in kWh.
Streamlit live URL::https://household-energy-consumption-forcasting-nxeengvhxw7tpcpfnztn6n.streamlit.app/

## Project Overview

This project is designed for household energy analysis and consumption forecasting. It provides:

- A dashboard for exploring historical electricity consumption.
- Dataset preview with household and daily usage records.
- KPI summaries for records, households, average consumption, temperature, and AC adoption.
- Interactive charts for trends, household size, AC usage, temperature, day of week, and feature correlation.
- A prediction page driven by the trained `household_energy_model.pkl` file.
- Input ranges and categorical options derived from the local dataset.
- Daily and estimated monthly consumption output.
- A comparison of the prediction with the historical dataset average.

## Application Pages

### Dashboard

The dashboard loads `household_energy_consumption.csv` and presents:

- Total records and unique households.
- Average daily energy consumption in kWh.
- Average temperature and percentage of homes with AC.
- A historical consumption trend.
- Consumption distribution by household size.
- AC versus non-AC consumption comparison.
- Temperature versus consumption relationship.
- Average consumption by weekday.
- Numeric feature correlation heatmap.
- A searchable and scrollable dataset preview.

### Predict Consumption

Users can provide the same household and calendar information used by the model:

1. Select a household ID for reference.
2. Choose household size.
3. Select whether the home has air conditioning.
4. Set average temperature.
5. Set peak-hours usage.
6. Select a prediction date.
7. Generate the predicted daily energy consumption.

The app automatically derives `Year`, `Month`, `Day`, `DayOfWeek`, and `IsWeekend` from the selected date.

### About

The About page documents the project purpose, model features, and application workflow.

## Dataset

The application uses `household_energy_consumption.csv`. Each row represents one household’s daily electricity usage.

| Column | Description |
| --- | --- |
| `Household_ID` | Unique household identifier. |
| `Date` | Daily observation date. |
| `Energy_Consumption_kWh` | Target value: daily energy consumption in kilowatt-hours. |
| `Household_Size` | Number of people in the household. |
| `Avg_Temperature_C` | Average temperature for the observation day in Celsius. |
| `Has_AC` | Whether the household has air conditioning: `Yes` or `No`. |
| `Peak_Hours_Usage_kWh` | Energy used during peak hours in kWh. |

The app engineers these calendar features from `Date` when they are not already present:

- `Year`
- `Month`
- `Day`
- `DayOfWeek` where Monday is `0`
- `IsWeekend` where weekend is `1` and weekday is `0`

## Model Features

The prediction pipeline receives features in this exact order:

```text
Household_Size
Avg_Temperature_C
Has_AC
Peak_Hours_Usage_kWh
Year
Month
Day
DayOfWeek
IsWeekend
```

`Has_AC` is passed as the original `Yes`/`No` text value so it remains compatible with the categorical preprocessing used during training.

## Repository Structure

```text
Household-Energy-consumption-forcasting/
├── app.py                              # Streamlit application
├── household_energy_consumption.csv    # Historical household dataset
├── household_energy_model.pkl         # Trained prediction pipeline
├── Household_energy_consumption.ipynb  # Exploration and model training notebook
├── requirements.txt                    # Deployment dependencies
├── Requerments.txt                     # Legacy dependency list
└── README.md                           # Project documentation
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/chintanthesiya/Household-Energy-consumption-forcasting.git
cd Household-Energy-consumption-forcasting
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

The app loads the model and CSV automatically from the same folder as `app.py`. The model is intentionally not uploaded through the browser; this avoids corrupted large-file uploads and keeps the deployment workflow predictable.

## Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Create a new app in [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Select the repository and the `main` branch.
4. Set the main file to `app.py`.
5. Deploy.

Streamlit Cloud detects `requirements.txt` automatically. Keep these files in the repository root:

- `app.py`
- `requirements.txt`
- `household_energy_consumption.csv`
- `household_energy_model.pkl`

The deployed Python environment should use compatible versions of Python, scikit-learn, XGBoost, and the libraries used when the model was saved. If the pickle cannot be loaded, re-export it with `joblib.dump()` from a compatible training environment.

## Use Cases

- Household electricity usage benchmarking.
- Daily consumption estimation for household planning.
- Understanding how AC usage, household size, temperature, and peak usage relate to energy demand.
- Demonstrating an end-to-end machine learning workflow from dataset exploration to deployment.
- Building a foundation for energy-efficiency recommendations and demand planning.

## Technology Stack

- Python
- Streamlit
- Pandas and NumPy
- Plotly
- Scikit-learn
- XGBoost-compatible trained pipeline
- Joblib/Pickle model loading

## Notes

This application is an estimation tool based on historical data. Predictions should be interpreted as model outputs rather than utility-billing guarantees. The quality of future predictions depends on the representativeness of the training data and compatibility between the saved model environment and deployment environment.

## License

No license has been specified for this repository yet.
