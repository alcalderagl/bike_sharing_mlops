import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    df = df.dropna().drop_duplicates()
    df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()