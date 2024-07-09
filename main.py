import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as model
from sklearn.metrics import (f1_score, recall_score, 
                             roc_auc_score, confusion_matrix, 
                             balanced_accuracy_score)




class Main():
    def __init__(self, file_path):
        # Excel dosyasını ',' ile ayırarak oku
        self.data_xlsx = pd.read_excel(file_path)
        if self.data_xlsx.shape[1] == 1:
            self.data = self.data_xlsx[self.data_xlsx.columns[0]].str.split(',', expand=True)
            self.data.columns = self.data_xlsx.columns[0].split(',')
    

    def change_column_type(self):
         # 'datetime' sütununu tarih/zaman formatına çevir
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        # Belirtilen sütunlar haricindeki tüm sütunları float tipine çevir
        float_columns = self.data.columns.difference(['failure', 'model', 'datetime'])
        self.data[float_columns] = self.data[float_columns].astype(float)


    def data_profiling(self):
        if self.data_xlsx is not None:
            # Veri profil raporu oluştur
            profile = ProfileReport(self.data, title="Pandas Profiling Report",)
            profile.to_file("data_profiling.html")
        else:
            print("Data is not loaded. Please load the data using read_data method first.")


    def is_null_and_duplicate(self):
        print(f"Null values: {self.data.isnull().sum().sum()}", f"Duplicate values: {self.data.duplicated().sum()}","NaN values: ",self.data.isna().sum().sum())
              
              
    def encoding(self):
        age_mapping = {2: 3,7: 2,8: 1,18: 0}
        self.data['age'] = self.data['age'].map(age_mapping)
        onehot_encoder = OneHotEncoder()
        self.data['model'] = onehot_encoder.fit_transform(self.data['model'].values.reshape(-1, 1)).toarray()
        self.data['failure'] = self.data['failure'].apply(lambda x: 1 if x == 'none' else 0)
       
         
    def split_date(self):
        self.data['day'] = self.data['datetime'].dt.day
        self.data['month'] = self.data['datetime'].dt.month
        self.data['year'] = self.data['datetime'].dt.year
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data.sort_values('datetime', inplace=True)
        self.data.set_index('datetime', drop=True, inplace=True)


    def cap_numerical_outliers_IQR(self):
        self.categorical_cols = ['machineID', 'model', 'age', 'error1count', 'error2count', 'error3count', 'error4count', 'error5count', 'failure']
        numerical_cols = self.data.columns.difference(self.categorical_cols)
        Q1 = self.data[numerical_cols].quantile(0.25)
        Q3 = self.data[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for col in numerical_cols:
            self.data[col] = np.where(self.data[col] < lower_bound[col], lower_bound[col], self.data[col])
            self.data[col] = np.where(self.data[col] > upper_bound[col], upper_bound[col], self.data[col])
  

    def scaled_data(self):
        scale_columns = self.data.columns.difference(self.categorical_cols)
        scaler = StandardScaler()
        self.data[scale_columns] = scaler.fit_transform(self.data[scale_columns])
        

    def create_time_window_features(self):
        # Calculate maximum, minimum, and average values for the past 48 hours
        time_windows = ['3h', '24h']
        metrics = ['min', 'max', 'mean', 'sd']
        features = ['volt', 'rotate', 'pressure', 'vibration']
        new_cols = {}  
        for window in time_windows:
            for feature in features:
                for metric in metrics:
                    column_name = f'{feature}_{metric}_{window}'
                    # Rolling window calculations
                    new_cols[f'{column_name}_max_48h'] = self.data[column_name].rolling('48h').max()
                    new_cols[f'{column_name}_min_48h'] = self.data[column_name].rolling('48h').min()
                    new_cols[f'{column_name}_mean_48h'] = self.data[column_name].rolling('48h').mean()
        new_cols_df = pd.DataFrame(new_cols, index=self.data.index)
        self.data = pd.concat([self.data, new_cols_df], axis=1)
        

    def split_data(self):
       
        self.X = self.data.drop('failure', axis=1)
        self.y = self.data['failure']
        # Strategy for random undersampling
        sampling_strategy_under = {1: 6000}
        # Strategy for SMOTE
        sampling_strategy_smote = {0: 100}
        # Random undersampling
        random_under_sampler = RandomUnderSampler(random_state=99, replacement=True, sampling_strategy=sampling_strategy_under)
        self.X, self.y = random_under_sampler.fit_resample(self.X, self.y)
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, stratify=self.y, random_state=11, shuffle=True
        )
        # SMOTE
        smote_k_neighbors = min(2, len(self.y_train[self.y_train == 1]) - 1)
        smote = SMOTE(random_state=8, sampling_strategy=sampling_strategy_smote, k_neighbors=smote_k_neighbors)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
      
    def draw_plot(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='volt_min_3h', y='rotate_min_3h', hue='failure', palette='viridis', s=50)
        plt.title('4 Sınıf İçin Dağılım Grafiği')
        plt.xlabel('X Ekseni')
        plt.ylabel('Y Ekseni')
        plt.legend(title='Sınıf')
        plt.grid(True)
        plt.show()


    def draw_cumulative_explained_variance(self):
        pca = PCA().fit(self.X_train)
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = explained_variance.cumsum()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()


    def apply_pca(self):
        pca = PCA(n_components=100)
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        

    def xgboost_model(self):
      weights = np.ones_like(self.y_train)
      weights[self.y_train == 0] = 100
      weights[self.y_train == 1] = 1
      self.model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        random_state=42  
    )
      self.model.fit(self.X_train, self.y_train, sample_weight=weights)


    def grid_search_xgboost(self):
        print(self.X_train.shape)
        print(self.y_train.shape)
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.05],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0]
        }

        weights = np.ones_like(self.y_train)
        weights[self.y_train == 0] = 100
        weights[self.y_train == 1] = 1

        grid_search = GridSearchCV(
            estimator=XGBClassifier(objective='binary:logistic'),
            param_grid=param_grid,
            n_jobs=-1,
            cv=3,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train, sample_weight=weights)

        print("En iyi parametreler:", grid_search.best_params_)
     

    def evaluate_model(self):
        y_predict = self.model.predict(self.X_test)
        print(
            f"F1 Score: {f1_score(self.y_test, y_predict):.4f}, "
            f"Recall Score: {recall_score(self.y_test, y_predict):.4f}, "
            f"ROC AUC Score: {roc_auc_score(self.y_test, y_predict):.4f},\n "
            f"Confusion Matrix:\n {confusion_matrix(self.y_test, y_predict)}, \n"
            f"Balanced Accuracy Score: {balanced_accuracy_score(self.y_test, y_predict):.4f}"
        )
        

    def model_save(self):
        model.dump(self.model, 'last_model.pkl')

    
    def load_model(self):
        self.model = model.load('last_model.pkl')



      