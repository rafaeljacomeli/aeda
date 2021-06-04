# %%
import pandas as pd
import sys
sys.path.insert(1, '../src')
from aeda import AEDA

# %%
# test
def main() -> None:
    """Sample code to test the AEDA functionality.
    """

    df = pd.read_csv('data_sample.csv', encoding='latin1')
    main_date = 'ORDERDATE'
    main_value = 'SALES'

    aeda = AEDA(df, main_date, main_value)
    aeda.start()

    print("Finished!")

# %%
if __name__ == '__main__':
    main()
# %%
