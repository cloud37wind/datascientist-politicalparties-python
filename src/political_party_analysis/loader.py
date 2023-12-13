from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import List

import pandas as pd


class DataLoader:
    """Class to load the political parties dataset"""

    local_data_path: str = Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"])

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        return pd.read_stata(self.local_data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        return df.drop_duplicates()

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        df_cols_removed = df.drop(non_features, axis=1)
        return df_cols_removed.set_index(index)

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe, 
        drop all NaN columns,
        use mean values of each column to fill NaN values."""
        mean_val = df.mean()
        return df.dropna(how='all', axis=1).fillna(mean_val)
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        scaler = StandardScaler()
        scaled_val = scaler.fit_transform(df)
        return pd.DataFrame(scaled_val, columns=df.columns, index=df.index)

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        df_deduplicated = self.remove_duplicates(self.party_data)
        df_non_removed = self.remove_nonfeature_cols(df_deduplicated, self.non_features, self.index)
        df_nan_handled = self.handle_NaN_values(df_non_removed)
        return self.scale_features(df_nan_handled)
