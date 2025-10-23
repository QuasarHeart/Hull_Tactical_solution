import numpy as np
import pandas as pd
import pandas.api.types

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Returns:
        float: The calculated adjusted Sharpe ratio.
    """

    if not pandas.api.types.is_numeric_dtype(submission['prediction']):#如果提交结果不是数字类型的，报错
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution#保存提交的数据，这里的solution好像是那个测试集
    solution['position'] = submission['prediction']#将提交的数据赋值给solution的position列，我觉得这里是新创建了一个列？并且把submission里的prediction列加进去
    #确保不要超出0-2的仓位限制
    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')
    #计算策略回报，但是这个公式为啥要这样写？？？
    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']
    #计算策略的夏普比率
    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']#计算超额收益，这个超额收益是策略回报减去单位风险，但是这个风险是怎么算出来的，虽然这个是测试集里本来有的
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()#这个prod函数是什么意思
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1#这个计算是干嘛的
    strategy_std = solution['strategy_returns'].std()#这个std函数怎么理解

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)#这一行是干嘛
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)#计算波动率，这里的std怎么理解
    #计算标准的真实数据下的超额收益
    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()#这里和上面的不同

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)#计算市场的波动率
    #市场波动率不能为零
    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')
    #计算波动率惩罚，波动率不能超过120%
    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol#波动率惩罚
    #这个公式为啥要这么写
    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,#这里应该是重要的一个优化点，要确保上面的仓位预测尽量准确，因为这里的策略超额收益是和仓位密切相关的
    )
    return_penalty = 1 + (return_gap**2) / 100#仓位的微小变化都会影响最终的分数

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)#计算最终的改进后的夏普比率
    return min(float(adjusted_sharpe), 1_000_000)