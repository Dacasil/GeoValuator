import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
from GeoValuator import DATA_DIR


### Configuration ###

CHECKPOINT_NAME = 'download_checkpoint_berlin.json'
normalized_mse = 0.0339


### Functions ###
def transform_mse_to_euros(normalized_mse, scaler, log_rent):
    """
    Transform MSE from normalized-log scale to approximate RMSE in Euros
    """
    # Step 1: Reverse MinMax scaling to get MSE on log scale
    min_log = scaler.data_min_[0]
    max_log = scaler.data_max_[0]
    range_log = max_log - min_log
    mse_log = normalized_mse * (range_log ** 2)
    
    # Convert to RMSE on log scale
    rmse_log = np.sqrt(mse_log)
    
    # Convert to approximate percentage error
    multiplicative_error = np.exp(rmse_log)
    
    # Convert to absolute error in Euros using geometric mean as reference
    geometric_mean_original = np.exp(np.mean(log_rent))
    rmse_euros_approx = geometric_mean_original * (multiplicative_error - 1)
    
    return rmse_euros_approx


### Code ###

file_path = os.path.join(DATA_DIR, 'interim', CHECKPOINT_NAME)

with open(file_path, 'r') as f:
    data = json.load(f)

rent_prices = {}
districts = []

for item in data['successful_downloads']:
    district = item['district']
    districts.append(district)
    rent_price = item['rent_price']
    rent_prices[district] = rent_price

rent_prices
districts = list(set(districts))

rent = []
for district in districts:
    rent.append(rent_prices[district])


log_rent = np.log1p(rent)
log_rent_2d = log_rent.reshape(-1, 1)
scaler = scaler = MinMaxScaler()
normalized_rent = scaler.fit_transform(log_rent_2d)

print(transform_mse_to_euros(normalized_mse, scaler, log_rent))