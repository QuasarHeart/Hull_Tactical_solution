"""Custom transformer for preprocessing data using Polars."""

import polars as pl

class PolarsTransformer:
    """A custom transformer for preprocessing data using Polars.

    This transformer provides functionality to scale, fill missing values, 
    and clip features in a DataFrame. It can fit on a given 
    dataset to compute statistics and apply transformations accordingly.

    Args:
        features (list, optional): List of feature columns to be transformed. Defaults to None.
        fillnull (bool, optional): Whether to fill null values with 0. Defaults to True.
        scale (bool, optional): Whether to scale the features 
                                by mean and standard deviation. Defaults to True.
        clip_time (bool, optional): Whether to clip the "feature_time_id" column 
                                    to its min and max values. Defaults to True.
    """
    def __init__(
        self,
        features: list = None,#传入要处理的特征列
        fillnull: bool = True,#判断是否是否要填充缺失值
        scale: bool = True,#归一化标准化处理
        #clip_time: bool = True#截断时间，如果在范围内出现时间上的异常值，那么取min
    ) -> None:
        """Initializes the PolarsTransformer class.

        Args:
            features (list, optional): List of feature columns to be transformed. Defaults to None.
            fillnull (bool, optional): Whether to fill null values with 0. Defaults to True.
            scale (bool, optional): Whether to scale the features 
                                    by mean and standard deviation. Defaults to True.这里是将特征均值标准化
            clip_time (bool, optional): Whether to clip the "feature_time_id" column 
                                        to its min and max values. Defaults to True.
        """
        self.features = features
        self.fillnull = fillnull
        self.scale = scale
        #self.clip_time = clip_time
        self.statistics_mean_std = None
        self.statistics_min_max = None

    def set_features(self, features: list) -> None:
        """Sets the list of features to be transformed.

        Args:
            features (list): List of feature columns to be transformed.
        """
        self.features = features

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fits the transformer on the given DataFrame and applies the transformations.

        Args:
            df (pl.DataFrame): The input Polars DataFrame.

        Returns:
            pl.DataFrame: The transformed Polars DataFrame.
        """
        #如果要标准化处理
        if self.scale:
            self.statistics_mean_std = {
                column: {
                    "mean": df[column].mean(),#为后续标准化计算全局均值
                    "std": df[column].std(),#为后续标准化计算全局标准差
                }
                for column in self.features
            }

        #if self.clip_time:
        #    self.statistics_min_max = {
        #        column: {
        #            "min": df[column].min(),
        #            "max": df[column].max(),
        #        }
        #        for column in ["feature_time_id"]#虽然只有一个列名但是仍然这么写是为了确保形式的一致性和可扩展性
        #    }

        if self.fillnull:
            df = df.with_columns([
                pl.col(column).fill_null(0.0)
                for column in self.features#将全部的缺失值填写为0.0
            ])
        #标准化处理
        if self.scale:
            df = df.with_columns([
                ((pl.col(column) - self.statistics_mean_std[column]["mean"]) /
                self.statistics_mean_std[column]["std"])
                for column in self.features
            ])

        return df

    def transform(self, df: pl.DataFrame, refit: bool = False) -> pl.DataFrame:
        """Applies the transformations to the given DataFrame using precomputed statistics.

        Args:
            df (pl.DataFrame): The input Polars DataFrame.
            refit (bool, optional): If True, updates the min and max values for 
                the "feature_time_id" column. Defaults to False.这里可以确定是否需要更新最大值和最小值，可以改变窗口大小

        Returns:
            pl.DataFrame: The transformed Polars DataFrame.
        """
        # if refit:
        #     if self.clip_time:
        #         self.statistics_min_max.update({
        #             column: {
        #                 "min": (
        #                     self.statistics_min_max[column]["min"]
        #                     if df[column].min() is None
        #                     else min(df[column].min(), self.statistics_min_max[column]["min"])
        #                 ),
        #                 "max": (
        #                     self.statistics_min_max[column]["max"]
        #                     if df[column].max() is None
        #                     else max(df[column].max(), self.statistics_min_max[column]["max"])
        #                 ),
        #             }
        #             for column in ["feature_time_id"]
        #         })

        # if self.clip_time:
        #     df = df.with_columns([
        #         pl.col(column).clip(
        #             self.statistics_min_max[column]["min"],
        #             self.statistics_min_max[column]["max"]
        #         )
        #         for column in ["feature_time_id"]
        #     ])

        if self.fillnull:
            df = df.with_columns([
                pl.col(column).fill_null(0.0)
                for column in self.features
            ])

        if self.scale:
            df = df.with_columns([
                ((pl.col(column) - self.statistics_mean_std[column]["mean"]) /
                    self.statistics_mean_std[column]["std"])
                for column in self.features
            ])

        return df
