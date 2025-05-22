# ARIMA + KF Residual

A lightweight hybrid that retains **ARIMA**’s transparency while auto-correcting its residuals with a gradient-trained **Kalman Filter**.

---

## ✨ Key Idea
1. **Automatic order selection** via the Box–Jenkins procedure.  
2. Treat the one-step-ahead residual as a **latent state**.  
3. Pass that state through a **learnable Kalman Filter** and add the filtered residual back to the linear forecast.


---

## 🏷️ Highlights
- **Plug-and-play**: drop-in replacement for vanilla ARIMA—minimal hyper-tuning.  
- **Fully interpretable**: keeps ARIMA coefficients and exposes learned \(Q\) and \(R\).  
- **Lean stack**: pure Python, `statsmodels`, and `torch`.  
- **Reproducible notebooks** included.

---

## 📂 Repository Structure

├── data/
│   ├── airlines2.txt   
│   ├── coloradoRiver.txt   
│   ├── lynx.txt
│	└── Sunspot.txt
├── datasets.py
├── ARIMAResidualKalmanFilter.py
└── README.md        



---

## 🚀 Installation
```bash
git https://github.com/mouglasgit/ARIMAResidualKalmanFilter.git
cd ARIMAResidualKalmanFilter
pip install -r requirements.txt
