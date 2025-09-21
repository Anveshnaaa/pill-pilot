PillPilot - AI-Powered Medicine Inventory Management System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Overview
Pill Pilot isn’t just another inventory management tool. Like most systems, it can track stock levels and provide clean dashboards. But what makes it different is the ability to go further — using AI to forecast demand and recommend store-to-store transfers. Instead of letting medicines expire in one location while another runs out, Pill Pilot balances inventory across pharmacies to reduce waste and ensure availability. The current test dataset covers 50 stores, but I have stress-tested the system up to 1,000 stores, demonstrating its ability to scale to much larger networks.


Quick Start


```bash
git clone https://github.com/Anveshnaaa/pill-pilot.git
cd pill-pilot
python3 run.py
```
The script will install everything and start the server automatically!
Then open http://localhost:5001 in your browser!


Key Features


1.Smart Transfer System(MOST IMPORTANT)
- Intelligent Recommendations: AI-driven transfer suggestions
- Cost Optimization: Minimize transfer costs while maximizing impact
- Urgency Scoring: Prioritize transfers based on risk and demand
- Geographic Analysis: Consider store locations and delivery time

2. Demand Forecasting → Predicts medicine usage trends.
3. Stockout Risk Analysis → Highlights where shortages may happen.
4. Expiry Tracking → Flags medicines nearing expiration.
5. Interactive Dashboard → Live charts powered by Plotly.

Tech Stack

Backend → Python, Flask
Data/ML → Pandas, NumPy, Scikit-learn
Frontend → HTML, CSS, JavaScript, Plotly
Storage → CSV (extendable to SQL/NoSQL)

Sample Usage

Upload a CSV file with store inventory.
See live dashboards showing current stock + expiry alerts.
Run demand forecasting for the next 7–14 days.
View transfer suggestions to balance supply across stores.

Machine Learning Models

The system includes pre-trained models for:
- Time Series Forecasting: ARIMA, Exponential Smoothing
- Demand Prediction: Linear Regression, Random Forest
- Risk Assessment: Classification models for stockout prediction
- Transfer Optimization: Clustering and optimization algorithms



## 🏗️ Project Structure

```
pillpilot-inventory-management/
├── app.py                          # Main Flask application
├── demand_forecasting_model.py     # ML models and forecasting
├── run.py                          # Application startup script
├── requirements.txt                # Python dependencies
├── sample_inventory.csv           # Sample data for testing
├── enterprise_inventory.csv       # Larger dataset example
├── templates/                      # HTML templates
│   ├── index.html                 # Main dashboard
│   ├── transfer.html              # Transfer management
│   └── analytics.html             # Analytics dashboard
├── static/                        # CSS and JavaScript files
│   ├── style.css                  # Styling
│   └── script.js                  # Frontend logic
├── ml_models/                     # Pre-trained ML models
│   └── demand_models.pkl          # Serialized models
└── README.md                      # This file
```
<img width="1440" height="857" alt="Screenshot 2025-09-21 at 12 29 37 AM" src="https://github.com/user-attachments/assets/a9851949-672a-4ae0-b4d5-8d2cd39829e8" />

<img width="1440" height="857" alt="Screenshot 2025-09-21 at 12 29 48 AM" src="https://github.com/user-attachments/assets/246f0f8f-7d01-4d1c-a6fa-f617957344dc" />

<img width="1440" height="857" alt="Screenshot 2025-09-21 at 12 30 28 AM" src="https://github.com/user-attachments/assets/24421890-b17e-47cc-bd54-ae787aa55257" />

<img width="1440" height="857" alt="Screenshot 2025-09-21 at 12 31 24 AM" src="https://github.com/user-attachments/assets/56c2448b-5cdb-42bc-8e0f-9dcb3a063ca3" />

<img width="1440" height="857" alt="Screenshot 2025-09-21 at 12 31 28 AM" src="https://github.com/user-attachments/assets/26b639d1-93dc-4e57-8a42-866abc60be30" />

<img width="1440" height="857" alt="Screenshot 2025-09-21 at 12 31 35 AM" src="https://github.com/user-attachments/assets/dbd48e9f-1f96-4cf9-bb39-ef912a929977" />

<img width="1440" height="857" alt="Screenshot 2025-09-21 at 12 31 39 AM" src="https://github.com/user-attachments/assets/3d5fad6c-f278-49ae-a1a1-d4574cd4744d" />







