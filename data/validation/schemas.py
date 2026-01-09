import pandera.pandas as pa
from pandera.typing import Series
from datetime import datetime

class TickSchema(pa.DataFrameModel):
    """Strict schema for Tick data."""
    epoch: Series[int] = pa.Field(ge=0, coerce=True)
    quote: Series[float] = pa.Field(gt=0, coerce=True)
    
    class Config:
        strict = True
        coerce = True

class CandleSchema(pa.DataFrameModel):
    """Strict schema for Candle data per SSOT Section 5.2.1."""
    epoch: Series[int] = pa.Field(ge=0, coerce=True)
    open: Series[float] = pa.Field(gt=0, coerce=True)
    high: Series[float] = pa.Field(gt=0, coerce=True)
    low: Series[float] = pa.Field(gt=0, coerce=True)
    close: Series[float] = pa.Field(gt=0, coerce=True)
    volume: Series[float] = pa.Field(ge=0, coerce=True, nullable=True)
    
    @pa.dataframe_check
    def high_gte_low(cls, df) -> Series[bool]:
        """Ensure High is greater than or equal to Low (SSOT validation)."""
        return df["high"] >= df["low"]
    
    class Config:
        strict = True
        coerce = True
