import math
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import datasets as data
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


class KalmanFilter(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.log_Q = nn.Parameter(torch.zeros(1))  # stores the noise 
        self.log_R = nn.Parameter(torch.zeros(1))  # noise measurement

    def forward(self, z):
        Q = torch.exp(self.log_Q)
        R = torch.exp(self.log_R)

        x = z[:, :, 0].unsqueeze(-1)           # initial state 
        P = (Q + R) * torch.ones_like(x)       # initial covariance

        x_f = torch.zeros_like(z)
        x_f[:, :, 0] = x.squeeze(-1)

        for k in range(1, self.window_size):
            x_pred = x
            P_pred = P + Q  # prediction

            K = P_pred / (P_pred + R) ## Kalman gain
            residual = z[:, :, k] - x_pred.squeeze(-1) #correction
            x = x_pred + K * residual.unsqueeze(-1)
            P = (1 - K) * P_pred

            x_f[:, :, k] = x.squeeze(-1)

        return x_f

class Arima:
    def __init__(self, train, test, order=(1,1,1)):
        self.train=train
        self.test=test
        self.order=order
        
        self.history = self.train.tolist()
        self.preds   = []

    def forecast(self):
        for t in range(len(self.test)):
            model = ARIMA(self.history, order=self.order).fit()
            yhat  = model.forecast()[0]
            self.preds.append(yhat)
            self.history.append(self.test[t])
        return np.array(self.preds)


class ArimaResidualKF:
    def __init__(self):
        
        parser = argparse.ArgumentParser(description='Parameters for selecting the serie.')
        parser.add_argument(
            '--base', 
            type=int, 
            default=0, 
            help='''Select the database:
            0: airlines
            1: colorado river
            2: sunspot
            3: lynx
            Padrão: %(default)s'''
        )
        args = parser.parse_args()
        
        base = int(args.base)

        choice_base={
            0: "airlines",
            1: "colorado_r",
            2: "sunspot",
            3: "lynx"
        }
        
        train, test, params=self.select_dataset_params(choice_base[base])
        
        preds_arima = Arima(train, test, params).forecast()
    
        resid_test = self.calculate_residue(preds_arima, test)
        
        resid_filt = self.filters_only_residue(resid_test)
        
        preds_corrected = self.add_filtered_residual_forecast(resid_filt, preds_arima)
    
        
        print(">> Only ARIMA:")
        self.model_metrics(test,      preds_arima, "onlyArima")
    
        print(">> ARIMA + Kalman on the residual:")
        self.model_metrics(test, preds_corrected, "arimaResidualKalman")


    def calibrate_arima(self, train):
        arima_auto = auto_arima(train, seasonal=False, stepwise=True,
                                max_p=20, max_q=20, max_d=1, trace=False,
                                error_action='ignore', suppress_warnings=True)
        p, d, q = arima_auto.order
        order       = (p, d, q)
        print(f"Parameters chosen for ARIMA({p},{d},{q})")
    
    
    def calculate_residue(self, preds_arima, test):
        resid_test = test - preds_arima
        return resid_test
    
    def filters_only_residue(self, resid_test):
        device = 'cpu'
        kf = KalmanFilter(window_size=len(resid_test)).to(device)
        optimizer = torch.optim.Adam(kf.parameters(), lr=1e-2)
        criterion = nn.MSELoss()
        
        z  = torch.from_numpy(resid_test).view(1,1,-1).double().to(device)
                
        for epoch in range(10):
            optimizer.zero_grad()
            x_f = kf(z)
            loss = criterion(x_f, z)
            loss.backward()
            optimizer.step()
            
        resid_filt = kf(z).detach().cpu().numpy().reshape(-1)
            
        return resid_filt
    
    def add_filtered_residual_forecast(self, resid_filt, preds_arima):
        preds_corrected = preds_arima + resid_filt
        return preds_corrected
    
    def select_dataset_params(self, choice_base):
        print(f"Chosen series: {choice_base}")
        
        if choice_base=="airlines":
            train, test = data.load_series_airlines()
            params       = (15, 2, 2)
            
        elif choice_base=="colorado_r":
            train, test = data.load_dataset_conv_cr()
            params       = (15, 2, 2)
        
        elif choice_base=="sunspot":
            train, test = data.load_dataset_conv_sun()
            params       = (15, 2, 2)
            
        elif choice_base=="lynx":
            train, test = data.load_dataset_conv_lynx()
            params       = (2, 0, 0)
        
                
        self.calibrate_arima(train)
        return train, test, params


    def save_data(self, y_true, y_pred, name):
        df = pd.DataFrame({
            'observed': y_true,
            'predicted'  : y_pred
        })
        
        df.to_csv(f"{name}.csv", index=False, float_format='%.6f')
        print(f"File {name}.csv saved successfully.")
    
    def model_metrics(self, y_true, y_pred, name):
        
        
        self.save_data(y_true, y_pred, name)
        
        mse  = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        mae  = mean_absolute_error(y_true, y_pred)
        mask = y_true != 0
        mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100
        evs = explained_variance_score(y_true, y_pred)
        r2  = r2_score(y_true, y_pred)
        
        print(f"MSE:  {mse:.8f}")
        print(f"RMSE: {rmse:.8f}")
        print(f"MAE:  {mae:.8f}")
        print(f"MAPE: {mape:.8f}%")
        print(f"EVS:   {evs:.8f}")
        print(f"R2:     {r2:.8f}")
        
        print(f"{mse:.6f} & {rmse:.6f} & {mae:.6f} & {mape:.6f} & {evs:.6f} & {r2:.6f}") 
        
        plt.figure(figsize=(10,4))
        plt.plot(y_true, label='True')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        output_path = f"{name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()
        
if __name__ == "__main__":
    ArimaResidualKF()
    