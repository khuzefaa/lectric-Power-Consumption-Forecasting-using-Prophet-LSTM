# lectric-Power-Consumption-Forecasting-using-Prophet-LSTM
A time series forecasting project predicting household electric power consumption using historical data from the UCI Machine Learning Repository. This project implements both Facebook Prophet for trend/seasonality modeling and a lightweight LSTM neural network for sequential prediction.
# Electric Power Consumption Forecasting (Prophet + LSTM)

## 📌 Overview
This project predicts **daily household electric power consumption** using two approaches:
1. **Facebook Prophet** – for capturing trends & seasonality.
2. **Lightweight LSTM Neural Network** – for sequential forecasting.

The dataset comes from the **UCI Machine Learning Repository** (Household Power Consumption Dataset).  
The notebook is optimized to run **entirely on CPU**, making it compatible with **free Google Colab** sessions without GPU initialization issues.

---

## 📊 Dataset
- **Source**: [UCI ML Repository - Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
- **Size**: ~2 million rows (minute-level data from 2006 to 2010)  
- **Features used**:
  - `Global_active_power` – total active power consumed (kW)
  - Date & Time converted into `Datetime`

---

## ⚙️ Steps Implemented
1. **Data Download** – via `kagglehub`
2. **Data Cleaning** – handle missing values, type conversions
3. **Resampling** – convert minute-level to **daily averages**
4. **Exploratory Data Analysis** – trend visualization
5. **Prophet Model** – forecast next 30 days
6. **LSTM Model** – sequence-based neural forecasting
7. **Visualization** – compare actual vs. predicted consumption

---

## 🛠 Installation & Usage

### **1️⃣ Install dependencies**
```bash
pip install kagglehub prophet matplotlib pandas numpy tensorflow
