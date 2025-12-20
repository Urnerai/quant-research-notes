from typing import Union
import pandas as pd
import numpy as np

def compute_log_returns(
        price: Union[pd.Series, pd.DataFrame]
) -> pd.Series:
    """
    Computes log returns from price data.

    Parameters
    ----------
    price : Union[pd.Series, pd.DataFrame]
        Price data.

    Returns
    -------
    pd.Series
        Time indexed log returns.
        ln(price_t / price_{t-1})
    """
    if isinstance(price, pd.DataFrame):
        if price.shape[1]!=1:
            raise ValueError("DataFrame must have exactly one column for price data.")
        price = price.loc[:,"price"]
    if not isinstance(price.index, pd.DatetimeIndex):
        raise ValueError("Price index must be a DatetimeIndex.")
    price=price.where(price>0)

    log_price=np.log(price)
    log_returns=log_price.diff()

    return log_returns
        