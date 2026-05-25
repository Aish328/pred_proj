# from pyexpat import features

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

class SmartGridDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        # self.data_hourly = None
        self.scaler = None

    def load_data(self):
        self.df = pd.read_csv(
            self.filepath,
            sep=',',
            # low_memory=False,
            encoding='latin1',
            engine='python',
            na_values=['?'],
            on_bad_lines='skip'
        )
        self.df.columns = self.df.columns.str.strip()
        print("\nColumns in Dataset:\n", self.df.columns.tolist())
        print(self.df.columns)
        return self.df

    def preprocess(self):
        # self.df.columns = self.df.columns.str.strip()   #remmoves extra spaces from column names

        self.df['Time'] = pd.to_datetime(
            self.df['Time'] ,
          
            errors='coerce'
        )                                                  #creates datetime column , error coerece: invalid missing values -> NaN 

        self.df = self.df.dropna(subset=['Time']) #removes rows where datetime conversion failed
        self.df.set_index('Time', inplace=True)# makes Time as index for time-series operations

        # self.df.drop(['Date', 'Time'], axis=1, inplace=True)
        
        features = [
                    'IR',
                    'IY',
                    'IB',
                    'VRY',
                    'VYB',
                    'VBR',
                    
                    'Active Load'
                ]
        # self.df['hour'] = self.df.index.hour
        # self.df['day'] = self.df.index.dayofweek
        for col in features:

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.extract(r'([-+]?\d*\.?\d+)')[0]
                .astype(float)
            )

        self.df = self.df.ffill() #fill : forward fill. fills missing values using previous values, good for time series continuity
        for col in features:

            q1 = self.df[col].quantile(0.01)
            q99 = self.df[col].quantile(0.99)

            self.df[col] = self.df[col].clip(q1, q99)

        # ---------------------------------------------
        # Add time-based features
        # ---------------------------------------------

        self.df['hour'] = self.df.index.hour
        self.df['day'] = self.df.index.dayofweek

        
        return self.df

    # def resample_hourly(self):      #groups data into 1 hour interval, takes mean of values in each hourr
    #     self.data_hourly = self.df.resample('h').mean()
    #     self.data_hourly['hour'] = self.data_hourly.index.hour
    #     self.data_hourly['day'] = self.data_hourly.index.dayofweek

        
    #     return self.data_hourly

    def scale_data(self):       #only selectedcolumns are used for training
        features = [
            'IR', 'IY', 'IB', 'VRY', 'VYB', 'VBR', 'Active Load','hour','day'
        ]
        # self.data_hourly =  self.data_hourly.dropna(subset=features) #ensure clean input for scaling
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(self.df[features])      #converts all features to range [0,1] for nn to train faster, avoids dominance f large values

        return scaled   #returns numpy not dataframe
    def generate_fault_labels(self,data):
        labels = np.zeros(len(data))

        for i in range(len(data)):
            row = data[i]
            IR = row[0]
            IY = row[1] 
            IB = row[2]
            VRY = row[3]
            VYB = row[4]    
            VBR = row[5]
            LOAD = row[6]
            if VRY < 0.20:
                labels[i] = 1

            # ==================================
            # 2 = CURRENT IMBALANCE
            # ==================================
            elif abs(IR - IY) > 0.40:
                labels[i] = 2

            # ==================================
            # 3 = OVERLOAD
            # ==================================
            elif LOAD > 0.80:
                labels[i] = 3

            # ==================================
            # 4 = VOLTAGE SWELL
            # ==================================
            elif VRY > 0.95:
                labels[i] = 4

            # ==================================
            # 5 = TRANSFORMER STRESS
            # ==================================
            elif LOAD > 0.70 and IR > 0.70:
                labels[i] = 5

        return labels.astype(int)


    def create_sequences(self, data, labels, seq_length=48):        #for time series forecasting : creates input output pairs for sliding window
        X = []
        y = []

        for i in range(len(data) - seq_length): #SLIDES WINDOW ACROSS DATASET
            X.append(data[i:i+seq_length])  #TAKES CHUNK OF 48 TIME STEPS
            y.append(labels[i+seq_length])  # assigns the corresponding fault label to the end of the sequence  

        return np.array(X), np.array(y)
    def get_processed_data(self, seq_length=48):

        self.load_data()

        self.preprocess()

        scaled_data = self.scale_data()
        labels = self.generate_fault_labels(scaled_data)

        X, _ = self.create_sequences(
            scaled_data,
            labels,
            seq_length
        )

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        return X