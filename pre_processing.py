import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class PreProcessing:

    def __init__(self, filepath: str, variance: float = 0.90, similarity_threshold: float = 0.85):
        self.filepath = filepath
        self.variance = variance
        self.threshold = similarity_threshold

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Returns a Dataframe via loading the csv from the given filepath."""

        return pd.read_csv(filepath)

    def get_pca_df(self, df: pd.DataFrame, variance: float) -> pd.DataFrame:
        """Returns DataFrame with data filtered with PCA."""

        pca = PCA(variance)
        pca_data = pca.fit_transform(df.values)
        pca_df = pd.DataFrame(pca_data)
        print(f'Applied PCA to each feature and merged all resulting DataFrame. \n Resulting PCA '
              f'DataFrame shape is `{pca_df.shape}`.')

        return pca_df

    def drop_similar_columns(self, df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
        """Returns a DataFrame with filtered columns by the similarity basis."""
        # Create correlation matrix
        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater or equal to threshold
        to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]

        # Drop features
        df.drop(to_drop, axis=1, inplace=True)
        print(f'Dropping similar columns if the similarity threshold is greater or equal to `{self.threshold}`.'
              f'\n Resulting DataFrame shape is: `{df.shape}`.')

        return df

    def scaling_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with scaled data."""

        # rounding off the values as we expect our data to be in int.
        columns = list(df.columns)
        df[columns] = df[columns].applymap(np.round)

        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame()
        scaled_df[list(df.columns)] = scaler.fit_transform(df[list(df.columns)])

        print(f'Applied MinMaxScaler to the DataFrame.')
        return scaled_df

    def __call__(self):
        df = self.load_csv(filepath=self.filepath)
        # asserts dataframe does not contain `NaN` values.
        assert not df.isnull().values.any()
        scaled_df = self.scaling_dataframe(df=df)
        filter_df = self.drop_similar_columns(df=scaled_df, threshold=self.threshold)
        pca_df = self.get_pca_df(df=filter_df, variance=self.variance)

        print('Finished preprocessing of data.')
        return pca_df
