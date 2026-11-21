"""Custom transformers for prepossessing data using Polars."""
import polars as pl

class PolarsTransformer:
    """a custom transformer for prepocessing data using Polars
    This transformer provides functionality to scale, fill missing values.
    It can fit on a given dataset to compute statistics and apply transformations accordingly.

    Args:
        features (list, optional): List of feature columns to be transformed. Defaults to None.
        fillnull (bool, optional): Whether to fill null values with 0. Defaults to True.
        scale (bool, optional): Whether to scale the features by mean and standard deviation. Defaults to True.
    """

    def __init__(
        self,
        features: list = None,
        fillnull: bool = True,
        scale: bool = True,
    )->None:
        """Initializes the PolarsTransformer class.

        Args:
            features (list, optional): List of feature columns to be transformed. Defaults to None.
            fillnull (bool, optional): Whether to fill null values with 0. Defaults to True.
            scale (bool, optional): Whether to scale the features by mean and standard deviation. Defaults to True.
        """

        self.features = features
        self.fillnull = fillnull
        self.scale = scale
        self.statistics_mean = None
        self.statistics_std = None
        self.statistics_median = None

    def set_features(self,features: list)->None:
        """
        Sets the list of features to be transformed.

        Args:
            features (list): List of features columns to be transformed.
        """
        self.features = features

    def fit_transform(self,df:pl.DataFrame)->pl.DataFrame:
        """Fits the transformer on the given DataFrame and applies the transformations.

        Args:
            df (pl.DataFrame): The input Polars DataFrame.

        Returns:
            pl.DataFrame: The transformed Polars DataFrame.
        """
        if self.scale:
            self.statistics_mean = [df[column].mean() for column in self.features]
            self.statistics_std = [df[column].std() for column in self.features]
            self.statistics_median = [df[column].median() for column in self.features]
        if self.fillnull:
            df.with_columns(
                pl.col(self.features).fill_null(self.statistics_median if self.statistics_median is not None else 0.0)
            )
        if self.scale:
            df = df.with_columns(
                (pl.col(self.features)-pl.lit(self.statistics_mean))/pl.lit(self.statistics_std)
            )

        return df

    
    def transform(self,df:pl.DataFrame)->pl.DataFrame:
        """Applies the transformations to the given DataFrame using precomputed statistics.

        Args:
            df (pl.DataFrame): The input Polars DataFrame.

        Returns:
            pl.DataFrame: The transformed Polars DataFrame.
        """
        if self.fillnull:
            df = df.with_columns(
                pl.col(self.features).fill_null(self.statistics_median if self.statistics_median is not None else 0.0)
            )
        if self.scale:
            df = df.with_columns(
                (pl.col(self.features)-pl.lit(self.statistics_mean))/pl.lit(self.statistics_std)
            )


        return df

        



        