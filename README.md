# ARIMA + KF Residual

A lightweight hybrid that retains **ARIMA**'s transparency while auto-correcting its residuals with a gradient-trained **Kalman Filter**.

---

## ✨ Key Idea
1. **Automatic order selection** via the Box–Jenkins procedure  
2. Treat the one-step-ahead residual as a **latent state**  
3. Pass that state through a **learnable Kalman Filter** and add the filtered residual back to the linear forecast

---

## 🏷️ Highlights
- **Plug-and-play**: drop-in replacement for vanilla ARIMA—minimal hyper-tuning  
- **Fully interpretable**: keeps ARIMA coefficients and exposes learned \(Q\) and \(R\)  
- **Lean stack**: pure Python, `statsmodels`, and `torch`  
- **Reproducible notebooks** included

---

## 📂 Repository Structure

project/
├── data/
│ ├── airlines2.txt
│ ├── coloradoRiver.txt
│ ├── lynx.txt
│ └── Sunspot.txt
├── ARIMAResidualKalmanFilter.py
├── datasets.py
├── requirements.txt
└── README.md


---

## 🚀 Installation
```bash
git clone https://github.com/mouglasgit/ARIMAResidualKalmanFilter.git
cd ARIMAResidualKalmanFilter
pip install -r requirements.txt


## 🧠 Core Components
ARIMAResidualKalmanFilter.py
Main implementation of the hybrid ARIMA-Kalman Filter model
Features:

Automatic ARIMA order selection

Differentiable Kalman Filter layer

Residual correction mechanism

datasets.py
Handles data loading and preprocessing for the 4 benchmark time series:

Supported Datasets
ID	Name	Description
0	airlines	Monthly airline passengers
1	colorado_r	Colorado River flow
2	sunspot	Sunspot activity
3	lynx	Canadian lynx trappings


## ⚙️ Command Line Interface
```bash
python ARIMAResidualKalmanFilter.py --base 2

--base    Time series dataset (0:airlines, 1:colorado_r, 2:sunspot, 3:lynx)


## 📈 Sample Workflow

Preprocess data using datasets.py

Initialize hybrid model from ARIMAResidualKalmanFilter.py

Automatically select ARIMA orders

Train Kalman Filter on residuals

Generate forecasts with residual correction

Visualize results and inspect Q/R parameters
