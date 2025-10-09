import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_path", required=True)
    parser.add_argument("--out", dest="model_path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    if "target" not in df.columns:
        df["target"] = df[df.columns[0]].rank(method="dense").astype(float)

    X = df[[c for c in df.columns if c != "target"]].select_dtypes(include=["number"]).fillna(0)
    y = df["target"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)

    joblib.dump(model, args.model_path)

if __name__ == "__main__":
    main()