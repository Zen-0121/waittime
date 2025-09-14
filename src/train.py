import pandas as pd
from sklearn.linear_model import LogisticRegression

# 入出力
TRAIN_CSV = "data/raw/train.csv"
TEST_CSV  = "data/raw/test.csv"
OUT_CSV   = "submission/submission.csv"

def main():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    # 最小の前処理（性別のみ）
    train["Sex"] = train["Sex"].map({"male":0, "female":1})
    test["Sex"]  = test["Sex"].map({"male":0, "female":1})

    X = train[["Sex"]]
    y = train["Survived"]

    model = LogisticRegression()
    model.fit(X, y)

    pred = model.predict(test[["Sex"]])
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred}) \
      .to_csv(OUT_CSV, index=False)

if __name__ == "__main__":
    main()
