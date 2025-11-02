"""Custom data processor for feature engineering and transformation.
"""
import polars as pl
from .config import PATHS_DATA, BASE_FEATURES, TO_LAGGED_FEATURES, T, FEATURES_TO_CREAT_WINDOWS
from .transformers import PolarsTransformer

class DataProcessor:
    def __init__(
            self,
            name: str = None, # 保持与您原来一致
            transformer: PolarsTransformer | None = None
    ):
        """
        initialize dataprocessor
        """
        # 【修改】我们不再在这里硬编码特征列表，因为它会动态生成
        self.features = None 
        self.transformer = transformer
        self.name = name
        
    def load_trainset(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Loads, engineers features, and splits data into train and validation sets
        in a leak-proof manner. Returns train_df, val_df, and raw val_df for scoring.
        """
        # 1. 加载足够大的数据范围
        df_full_raw = (
            pl.read_csv(PATHS_DATA["train"], try_parse_dates=False)
            .with_columns(pl.exclude('date_id').cast(pl.Float64, strict=False))
        )
        
        # 筛选出我们需要处理的总时间段
        df_to_process = df_full_raw.filter(pl.col('date_id') >= 6060)
        
        # 2. 在连续数据块上进行特征工程
        df_featured = self._add_features(df=df_to_process)
        
        # 3. 填充因计算产生的NaN值
        df_clean = df_featured.fill_null(strategy="forward").fill_null(strategy="backward")

        # 4. 在特征工程之后，进行分割
        train_df = df_clean.filter(pl.col('date_id') < 8700)
        val_df = df_clean.filter(
            (pl.col('date_id') >= 8700) & (pl.col('date_id') <= 8900)
        )
        
        # 额外分割出原始的验证集部分，用于打分
        val_df_raw = df_full_raw.filter(
            (pl.col('date_id') >= 8700) & (pl.col('date_id') <= 8900)
        )
        
        # 动态生成最终的特征列表
        self.features = BASE_FEATURES + self._get_new_feature_names()
        
        return train_df, val_df, val_df_raw

    def _add_features(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        add features
        """
        # 创建滞后特征
        df = df.with_columns(
            pl.col(col).shift(1).alias(f"lagged_{col}") for col in TO_LAGGED_FEATURES
        )
        # 创建滑动窗口特征
        for n in T:
            df = self._get_window_average_std(df=df, cols=FEATURES_TO_CREAT_WINDOWS, n=n)

        return df

    def _get_window_average_std(
        self,
        df: pl.DataFrame,
        cols: list,
        n: int 
    ) -> pl.DataFrame:
        # 【修复】将 with_columns 的结果重新赋值
        df = df.with_columns(
            pl.col(col).rolling_mean(window_size=n).alias(f"{col}_rolling_avg_{n}") for col in cols
        )
        df = df.with_columns(
            pl.col(col).rolling_std(window_size=n).alias(f"{col}_rolling_std_{n}") for col in cols
        )
        return df
        
    def _get_new_feature_names(self) -> list[str]:
        """Dynamically generates the names of the created features."""
        new_names = [f"lagged_{col}" for col in TO_LAGGED_FEATURES]
        for n in T:
            for col in FEATURES_TO_CREAT_WINDOWS:
                new_names.append(f"{col}_rolling_avg_{n}")
                new_names.append(f"{col}_rolling_std_{n}")
        return new_names