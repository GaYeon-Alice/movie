import pandas as pd
from settings import filename

def save_result(result, pred_all):
    result['prob'] = pred_all
    result['prob'] = round(result['prob'], 3)
    result.to_csv(filename)
    print('\nSaved Successfully!')