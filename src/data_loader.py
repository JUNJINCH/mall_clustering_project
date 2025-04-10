import pandas as pd
import logging

def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise