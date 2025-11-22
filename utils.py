import pandas as pd

def order_check(source:pd.DataFrame, stream:pd.DataFrame, target:str):
    source = source.drop(columns = target)
    stream = stream.drop(columns = target)
    order_cols = source.columns
    stream = stream[order_cols]
    return source, stream


def  num_cat_split(source:pd.DataFrame, stream:pd.DataFrame):
    source_num_cols=source.select_dtypes(include=['int','float'])
    source_cat_cols=source.select_dtypes(include=['object'])

    stream_num_cols=stream.select_dtypes(include=['int','float'])
    stream_cat_cols=stream.select_dtypes(include=['object'])

    return source_num_cols,source_cat_cols,stream_num_cols,stream_cat_cols


def main(source:pd.DataFrame, stream:pd.DataFrame, target:str):
    source,stream=order_check(source, stream, target)
    source_num_cols,source_cat_cols,stream_num_cols,stream_cat_cols=num_cat_split(source, stream)
    return source_num_cols,source_cat_cols,stream_num_cols,stream_cat_cols



