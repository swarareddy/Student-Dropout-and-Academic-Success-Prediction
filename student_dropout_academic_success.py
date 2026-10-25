"""Student Dropout and Academic Success analysis. Keep Studentdata.csv in the same folder."""
from IPython.display import display

# Predicting Student Dropout and Academic Success Using Data Science
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE = 42
DATA_PATH = "Studentdata.csv"  # Keep the CSV in the same folder as this notebook.
df = pd.read_csv(DATA_PATH, sep=";")
print("First five rows:")
display(df.head())
print("\nDataset shape:", df.shape)
print("\nData types:")
display(df.dtypes.astype(str).value_counts().rename_axis("dtype").to_frame("column_count"))
print("\nColumn names:")
print(df.columns.tolist())

# **Observation:** the file uses semicolons as delimiters. The target column is `Target`, with three classes: Dropout, Enrolled and Graduate. Numeric storage does not always mean a variable is truly continuous: many columns are coded categories (for example, application mode, course, nationality and occupations).
# 4. Data quality checks and variable types
target_col = "Target"
print("Target counts:")
display(df[target_col].value_counts().to_frame("count"))
print("\nTarget percentages:")
display((df[target_col].value_counts(normalize=True).mul(100).round(2)).to_frame("percent"))

missing = df.isna().sum().sort_values(ascending=False)
print("\nColumns with missing values:")
display(missing[missing > 0].to_frame("missing_count") if (missing > 0).any() else pd.DataFrame({"message":["No missing values detected."]}))

print("\nExact duplicate rows:", int(df.duplicated().sum()))
print("Unique target labels:", sorted(df[target_col].dropna().unique().tolist()))

binary_candidates = [c for c in df.columns if df[c].dropna().nunique() == 2 and c != target_col]
coded_categorical = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance\t", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation"
]
print("\nBinary-like columns (two distinct observed values):", binary_candidates)
print("\nExamples of coded categorical columns:", [c for c in coded_categorical if c in df.columns])

# Basic unusual-value checks. These are flags for review, not automatic deletions.
count_cols = [c for c in df.columns if "Curricular units" in c and any(k in c for k in ["credited","enrolled","evaluations","approved","without evaluations"])]
negative_counts = {c: int((df[c] < 0).sum()) for c in count_cols if pd.api.types.is_numeric_dtype(df[c])}
print("\nNegative curricular-unit counts by column:", negative_counts)
print("Age range:", (df["Age at enrollment"].min(), df["Age at enrollment"].max()))
print("Admission grade range:", (df["Admission grade"].min(), df["Admission grade"].max()))

# **Preprocessing decisions:** preserve valid rows; do not delete unusual values without checking their meaning. The pipeline below imputes missing numerical values with the median, scales numeric predictors, imputes categorical values with the most frequent category and one-hot encodes them. This avoids treating arbitrary category codes as ordered numeric quantities.
# 5. Exploratory data analysis and statistics
class_counts = df[target_col].value_counts().reindex(["Dropout", "Enrolled", "Graduate"])
class_percent = df[target_col].value_counts(normalize=True).reindex(["Dropout", "Enrolled", "Graduate"]) * 100
class_table = pd.DataFrame({"Count": class_counts, "Percent": class_percent.round(2)})
display(class_table)

plt.figure(figsize=(7, 5))
plt.bar(class_table.index, class_table["Count"])
plt.title("Distribution of Student Outcomes")
plt.xlabel("Final outcome")
plt.ylabel("Number of students")
plt.tight_layout()
plt.show()

print("Interpretation: the target classes are imbalanced. Accuracy alone may hide weak performance on the smaller Enrolled class.")

# Histograms of selected numerical attributes
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(df["Age at enrollment"].dropna(), bins=20, edgecolor="black")
axes[0].set_title("Age at Enrollment Distribution")
axes[0].set_xlabel("Age at enrollment (years)")
axes[0].set_ylabel("Number of students")

axes[1].hist(df["Admission grade"].dropna(), bins=20, edgecolor="black")
axes[1].set_title("Admission Grade Distribution")
axes[1].set_xlabel("Admission grade")
axes[1].set_ylabel("Number of students")
plt.tight_layout()
plt.show()

print("Descriptive statistics:")
display(df[["Age at enrollment", "Admission grade",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)"]].describe().round(2))

# Admission grade across final outcomes
outcome_order = ["Dropout", "Enrolled", "Graduate"]
plt.figure(figsize=(8, 5))
df.boxplot(column="Admission grade", by=target_col, grid=False)
plt.title("Admission Grades by Final Student Outcome")
plt.suptitle("")
plt.xlabel("Final outcome")
plt.ylabel("Admission grade")
plt.tight_layout()
plt.show()

admission_summary = df.groupby(target_col)["Admission grade"].agg(["count","mean","median","std"]).reindex(outcome_order).round(2)
display(admission_summary)
print("Interpretation: compare medians, spread and overlap. Differences are associations and do not prove admission grade causes an outcome.")

# First-semester academic performance by final outcome
plt.figure(figsize=(8, 5))
df.boxplot(column="Curricular units 1st sem (grade)", by=target_col, grid=False)
plt.title("First-Semester Grades by Final Student Outcome")
plt.suptitle("")
plt.xlabel("Final outcome")
plt.ylabel("First-semester grade")
plt.tight_layout()
plt.show()

sem1_summary = df.groupby(target_col)["Curricular units 1st sem (grade)"].agg(["count","mean","median","std"]).reindex(outcome_order).round(2)
display(sem1_summary)

# Age and outcome distribution
age_bins = pd.cut(df["Age at enrollment"], bins=[-np.inf,20,25,30,40,np.inf],
                  labels=["20 or younger","21–25","26–30","31–40","Over 40"])
age_outcome = pd.crosstab(age_bins, df[target_col], normalize="index").reindex(columns=outcome_order) * 100
display(age_outcome.round(1))
age_outcome.plot(kind="bar", stacked=True, figsize=(9,5))
plt.title("Outcome Composition by Age-at-Enrollment Group")
plt.xlabel("Age group")
plt.ylabel("Percentage within age group")
plt.legend(title="Final outcome")
plt.tight_layout()
plt.show()

print("Interpretation: semester grades may separate outcomes strongly because they occur during the course. They are useful for monitoring after semester one, but would leak future information in a model intended to predict at enrollment.")

# 6. Hypothesis test: first-semester grades across outcomes
grade_col = "Curricular units 1st sem (grade)"
groups = [df.loc[df[target_col] == label, grade_col].dropna().to_numpy()
          for label in outcome_order]
H_stat, p_value = kruskal(*groups)
alpha = 0.05
print("Kruskal–Wallis H statistic:", round(H_stat, 4))
print("p-value:", f"{p_value:.6g}")
print("Significance level:", alpha)
if p_value < alpha:
    print("Conclusion: reject H0. There is statistical evidence that first-semester grade distributions differ across at least two outcome groups.")
else:
    print("Conclusion: fail to reject H0. The test does not provide sufficient evidence of a difference across groups.")

# Approximate epsilon-squared effect size for Kruskal–Wallis
n_total = sum(len(g) for g in groups)
k = len(groups)
epsilon_sq = max(0, (H_stat - k + 1) / (n_total - k))
print("Approximate epsilon-squared effect size:", round(epsilon_sq, 4))
print("Interpretation: p-value measures evidence against H0; epsilon-squared helps describe the magnitude of group differences.")

# 7. Build an early-prediction multiclass logistic regression model
# Exclude target and all semester performance fields from early prediction.
post_enrollment_cols = [c for c in df.columns if "Curricular units 1st sem" in c or "Curricular units 2nd sem" in c]
X = df.drop(columns=[target_col] + post_enrollment_cols)
y = df[target_col].astype(str)

# Treat known code-based categories as categorical, even though they are stored as integers.
categorical_cols = [c for c in coded_categorical if c in X.columns]
# Binary indicators can remain numeric; one-hot encoding is used for coded multi-category variables.
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("Predictor count:", X.shape[1])
print("Excluded post-enrollment columns to prevent leakage:", len(post_enrollment_cols))
print("Categorical predictors:", categorical_cols)
print("Numerical/binary predictors:", numeric_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Training rows:", len(X_train), "| Test rows:", len(X_test))
print("Model trained and test predictions generated.")

# 8. Model evaluation
labels = ["Dropout", "Enrolled", "Graduate"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Confusion Matrix — Early-Prediction Logistic Regression")
plt.tight_layout()
plt.show()

report_dict = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
report = pd.DataFrame(report_dict).T
display(report.loc[labels + ["macro avg", "weighted avg"], ["precision","recall","f1-score","support"]].round(3))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Macro precision:", round(report_dict["macro avg"]["precision"], 4))
print("Macro recall:", round(report_dict["macro avg"]["recall"], 4))
print("Macro F1-score:", round(report_dict["macro avg"]["f1-score"], 4))

# Export metrics and confusion matrix for report/reproducibility.
report.to_csv("classification_report_metrics.csv")
pd.DataFrame(cm, index=labels, columns=labels).to_csv("confusion_matrix.csv")

# 9. Interpret model-derived feature information
fitted_preprocessor = model.named_steps["preprocess"]
fitted_lr = model.named_steps["classifier"]
feature_names = fitted_preprocessor.get_feature_names_out()
coef_df = pd.DataFrame(fitted_lr.coef_, columns=feature_names, index=fitted_lr.classes_)

for cls in fitted_lr.classes_:
    top_positive = coef_df.loc[cls].sort_values(ascending=False).head(8)
    top_negative = coef_df.loc[cls].sort_values().head(8)
    print(f"\n{cls}: largest positive coefficients")
    display(top_positive.to_frame("coefficient").round(3))
    print(f"{cls}: largest negative coefficients")
    display(top_negative.to_frame("coefficient").round(3))

print("Caution: coefficients describe model associations, not causal effects. Correlated predictors, coding choices and regularization affect coefficient sizes.")

# 10. Evidence-based recommendations