import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# 1. Fakana ny data (Capture 30 teo aloha)
# Ohatra: data = {'ora': ['14:05:01', '14:06:10', ...], 'multiplicateur': [1.5, 10.2, ...]}
def predict_aviator(history_data):
    df = pd.DataFrame(history_data)
    
    # Ovaina ho segondra ny ora mba ho azo kajiana
    df['ora_sec'] = df['ora'].apply(lambda x: sum(int(a) * 60**i for i, a in enumerate(reversed(x.split(":")))))
    
    # X dia ny ora sy ny filahany (index), y dia ny multiplicateur
    X = df[['ora_sec']]
    y = df['multiplicateur']
    
    # Famoronana ny Model (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.values, y.values)
    
    # Maminavina ny ora manaraka (ohatra 30 segondra aorian'ny farany)
    last_time = df['ora_sec'].iloc[-1]
    next_time = np.array([[last_time + 30]]) 
    
    prediction = model.predict(next_time)
    return prediction[0]

# Ohatra amin'ny fampiasana azy (Data 30 teo aloha)
data_30 = {
    'ora': ['10:00:01', '10:00:45', '10:01:15', '10:02:00'], # Tohizo hatramin'ny 30
    'multiplicateur': [1.20, 5.50, 1.05, 2.10] # Tohizo hatramin'ny 30
}

vokatra = predict_aviator(data_30)
print(f"Ny vinavina amin'ny manaraka dia manodidina ny: {vokatra:.2f}x")
