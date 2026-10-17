import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler   
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_predict
df = pd.read_csv("train.csv")

numeric_columns=["age","married","edu","children","asset_savings","asset_durable","fs_meat","fs_enoughtom","fs_sleephun"]

def prepare(df):
    """Sütunları sayıya çevirir (log dönüşümü yok)."""
    return df[numeric_columns].apply(pd.to_numeric, errors="coerce")


X = prepare(df)
y = df["depressed"]
 
model = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    StandardScaler(),
    LogisticRegression(max_iter=3000, C=0.05),
)
 

cv = StratifiedKFold(5, shuffle=True, random_state=0)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
model.fit(X, y)



def ask_number(prompt, lo, hi, integer=True):
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            val = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if not lo <= val <= hi:
            print(f"  Value must be between {lo} and {hi}.")
            continue
        return int(val) if integer else val
def ask_yes_no(prompt):
    while True:
        raw = input(prompt + " (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return 1
        if raw in ("n", "no"):
            return 0
        print("  Please type 'y' or 'n'.")
print("\n=== Depression Risk Estimate (for educational use) ===\n")
answers = {
    "age": ask_number("Your age: ", 17, 91),
    "married": ask_yes_no("Are you married?"),
    "edu": ask_number("Years of education (1-19): ", 1, 19),
    "children": ask_number("Number of children: ", 0, 10),
    "asset_savings": ask_number("Your savings (0 = none): ", 0, 1e9, integer=False),
    "asset_durable": ask_number("Value of your durable assets (0 = none): ", 0, 1e9, integer=False),
    "fs_meat": ask_number("How many times did you eat meat in the past week? ", 0, 22),
    "fs_enoughtom": ask_yes_no("Do you have enough food for tomorrow?"),
    "fs_sleephun": ask_yes_no("Have you gone to sleep hungry recently?"),
}

user = prepare(pd.DataFrame([answers]))
prob = model.predict_proba(user)[0, 1]
percentile = (oof < prob).mean() * 100

print(f"Estimated probability: {prob*100:.1f}%")
print(f"Higher than {percentile:.0f}% of people in the training data")