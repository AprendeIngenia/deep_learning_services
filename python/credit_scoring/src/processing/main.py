"""
Data Preprocessing Module for German Credit Risk Dataset
Handles feature engineering, encoding, and scaling following SOLID principles
"""
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class CreditDataPreprocessor:
    """
    Preprocessor for German Credit Risk dataset
    Handles categorical encoding, numerical scaling, and feature engineering
    """
    def __init__(self):
        self.numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
        self.categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        self.target_feature = 'Risk'
        
    def fit_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Builds a pipeline for preprocessing the data
        1. Scale numerical features (StandardScaler)
        2. Codify categorical features (OneHotEncoder)
        
        """
        
        # features transformers
        numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_tf = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        
        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_tf, self.numerical_features),
                ('cat', categorical_tf, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        # adjust
        x_train = df.drop(self.target_feature, axis=1)
        preprocessor.fit(x_train)
        return preprocessor
    
    def process_data(self, df: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply processing and separate features
        Args:
            df (pd.DataFrame): DataFrame to process.
            preprocessor (ColumnTransformer): fit preprocessing
        """
        
        df_copy = df.copy()
        df_copy[self.target_feature] = df_copy[self.target_feature].map({'bad': 0, 'good': 1})
        
        y = df_copy[self.target_feature]
        x = df_copy.drop(self.target_feature, axis=1)
        
        x_processed = preprocessor.transform(x)
        return x_processed, y