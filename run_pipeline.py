import numpy as np
import matplotlib.pyplot as plt
from src.model_lstm import train_lstm_model
from src.data_collection import get_race_data



df1 = get_race_data(2024, 1)
df2 = get_race_data(2024, 2)
df3 = get_race_data(2024, 3)

print(df1)
print(df2)
print(df3)