import pandas as pd
from pathlib import Path

def load_data(csv_name:str) -> pd.DataFrame:
    src_dir=Path(__file__).resolve().parent
    data_dir=src_dir.parent/"data"
    data_path=data_dir/csv_name


    df = pd.read_csv(data_path)

    df["Date"]=pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df=df[["Close"]].rename(columns={"Close":"price"})

    return df
def main():
    data = load_data("tsla.csv")
    print(data.head())

if __name__ == "__main__":
    main()
