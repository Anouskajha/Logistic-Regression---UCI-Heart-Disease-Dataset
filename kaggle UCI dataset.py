import pandas as pd
import numpy as np
from pathlib import Path
import sys 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, accuracy_score

def load_and_clean_data():
    # Define file path
    SRC = Path(r"C:\Users\Durba Jha\Downloads\heart_disease_uci.csv")
    
    #  Load the data
    print("Loading data...")
    df = pd.read_csv(SRC)
    
    #  Check missing values before cleaning
    print("\nMissing values in each column:")
    missing_counts = df.isnull().sum()
    for column in df.columns:
        count = missing_counts[column]
        if count > 0:
            print(f"{column}: {count} missing values")
    
    #  Handle missing values
    # For numeric columns: fill with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled {col} missing values with median: {median_value}")
    
    # For categorical columns: fill with mode (most common value)
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_value}")
    
    # Step 4: Verify no missing values remain
    print("\nRemaining missing values:")
    print(df.isnull().sum())
    
    # Step 5: Save cleaned data
    cleaned_file = SRC.parent / "heart_disease_clean.csv"
    df.to_csv(cleaned_file, index=False)
    print(f"\nSaved cleaned data to: {cleaned_file}")
    
    return df

def analyze_disease_burden(df):
    severity_dist = df['num'].value_counts().sort_index()
    severity_by_age = df.groupby('num')['age'].agg(['mean', 'std', 'count'])
    gender_severity = pd.crosstab(df['num'], df['sex'])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    severity_dist.plot(kind='bar')
    plt.title('Distribution of Heart Disease Severity')
    plt.xlabel('Disease Severity (0-4)')
    plt.ylabel('Number of Patients')

    plt.subplot(1, 2, 2)
    sns.boxplot(x='num', y='age', data=df)
    plt.title('Age Distribution by Disease Severity')
    plt.tight_layout()
    plt.show()

    return severity_by_age, gender_severity


def prepare_features(df, binary=True):
    df2 = df.copy()
    # convert boolean-like strings and sex
    df2 = df2.replace({'TRUE': 1, 'True': 1, 'FALSE': 0, 'False': 0})
    if 'sex' in df2.columns:
        df2['sex'] = df2['sex'].map({'Male': 1, 'Female': 0})
    # one-hot encode categorical columns commonly in this dataset
    cat_cols = [c for c in ['cp','restecg','slope','thal','dataset'] if c in df2.columns]
    if cat_cols:
        df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=True)
    # drop identifier
    if 'id' in df2.columns:
        df2 = df2.drop(columns=['id'])
    # target
    if binary:
        y = (pd.to_numeric(df2['num'], errors='coerce').fillna(0) > 0).astype(int)
    else:
        y = pd.to_numeric(df2['num'], errors='coerce').fillna(0).astype(int)
    X = df2.drop(columns=['num'])
    X = df2.drop(columns=['num'])
    # remove any 'dataset' column or one-hot 'dataset_*' dummies
    X = X.drop(columns=[col for col in X.columns if 'dataset' in col], errors='ignore')
    # ensure numeric; fill any remaining NA with median
    for c in X.columns:
        if X[c].dtype == object:
            try:
                X[c] = pd.to_numeric(X[c], errors='coerce')
            except Exception:
                X[c] = X[c].astype('category').cat.codes
        X[c].fillna(X[c].median(), inplace=True)
    return X, y

def run_logistic_regression(df, binary=True, scale=True):
    X, y = prepare_features(df, binary=binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    if binary:
        model = LogisticRegression(solver='liblinear', max_iter=1000)
    else:
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\nLogistic Regression Results")
    print("-" * 40)
    print(classification_report(y_test, preds))

    # confusion matrix plot 
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    out_path = Path.cwd() / "confusion_baseline.png"
    plt.savefig(out_path, bbox_inches='tight')
    print("Saved baseline confusion matrix to:", out_path)
    plt.show()

    if binary:
        try:
            probs = model.predict_proba(X_test)[:,1]
            auc_val = roc_auc_score(y_test, probs)
            print("ROC AUC:", auc_val)
            # small ROC plot saved
            fpr, tpr, _ = roc_curve(y_test, probs)
            plt.figure(figsize=(6,4))
            plt.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
            plt.plot([0,1],[0,1],'k--', alpha=0.6)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (baseline)')
            roc_path = Path.cwd() / "roc_baseline.png"
            plt.savefig(roc_path, bbox_inches='tight')
            print("Saved ROC plot to:", roc_path)
            plt.show()
        except Exception:
            pass
    return model
def print_confusion_metrics(y_true, y_pred):
    """Print TN/FP/FN/TP, accuracy, sensitivity, specificity and precision (safe for edge cases)."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except Exception:
        # fallback if confusion_matrix shape unexpected
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
        tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
        fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
        fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())

    total = tp + tn + fp + fn if (tp + tn + fp + fn) else 1
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity (recall, class=1): {sensitivity:.3f}")
    print(f"Specificity (class=0 recall): {specificity:.3f}")
    print(f"Precision (class=1): {precision:.3f}")

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": accuracy, "sensitivity": sensitivity,
            "specificity": specificity, "precision": precision}

def run_logistic_with_options(df, class_weight=None, resample_train=False, threshold=0.5):
    X, y = prepare_features(df, binary=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    if resample_train:
        train_df = pd.concat([X_train, y_train.rename('target')], axis=1)
        majority = train_df[train_df['target'] == train_df['target'].mode()[0]]
        minority = train_df[train_df['target'] != train_df['target'].mode()[0]]
        from sklearn.utils import resample
        minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        train_df = pd.concat([majority, minority_up])
        y_train = train_df['target']
        X_train = train_df.drop(columns=['target'])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(solver='liblinear', class_weight=class_weight, max_iter=2000)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    preds = (probs >= threshold).astype(int)
    print("\nResults with options:", "class_weight="+str(class_weight), "resample_train="+str(resample_train), "threshold="+str(threshold))
    print_confusion_metrics(y_test, preds)
    print("\nClassification report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (thr={threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    out_path = Path.cwd() / f"confusion_thr_{str(threshold).replace('.','_')}.png"
    plt.savefig(out_path, bbox_inches='tight')
    print("Saved confusion matrix to:", out_path)
    plt.show()

    fpr, tpr, thr = roc_curve(y_test, probs)
    youden = tpr - fpr
    best_idx = youden.argmax()
    best_thr = thr[best_idx]
    print(f"Suggested threshold by Youden's J: {best_thr:.3f} (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")
    return model, scaler

def plot_logistic_diagnostics(df, binary=True):
    X, y = prepare_features(df, binary=binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(solver='liblinear', max_iter=2000)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # ROC
    fpr, tpr, thr = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc_val:.3f}')
    plt.plot([0,1],[0,1],'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    roc_file = Path.cwd() / "roc_diagnostics.png"
    plt.savefig(roc_file, bbox_inches='tight')
    print("Saved ROC to:", roc_file)
    plt.show()

    # Probability histogram
    plt.figure(figsize=(6,4))
    plt.hist(probs[np.array(y_test)==0], bins=25, alpha=0.6, label='actual 0')
    plt.hist(probs[np.array(y_test)==1], bins=25, alpha=0.6, label='actual 1')
    plt.xlabel('Predicted probability (class=1)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Predicted probability distribution by true class')
    hist_file = Path.cwd() / "prob_hist_diagnostics.png"
    plt.savefig(hist_file, bbox_inches='tight')
    print("Saved probability histogram to:", hist_file)
    plt.show()

    # Coefficients (if features > 1)
    try:
        coefs = model.coef_.ravel()
        feat_names = X.columns
        coef_df = pd.Series(coefs, index=feat_names).sort_values()
        plt.figure(figsize=(8,6))
        coef_df.plot(kind='barh')
        plt.title('Logistic regression coefficients (log-odds)')
        plt.xlabel('Coefficient')
        plt.tight_layout()
        coef_file = Path.cwd() / "coefficients_diagnostics.png"
        plt.savefig(coef_file, bbox_inches='tight')
        print("Saved coefficients plot to:", coef_file)
        plt.show()
    except Exception:
        pass

    # Probability vs age (if age exists)
    if 'age' in X_test.columns:
        age_vals = X_test['age'].values
        plt.figure(figsize=(6,4))
        sns.regplot(x=age_vals, y=probs, lowess=False, scatter_kws={'s':10}, line_kws={'color':'red'})
        plt.xlabel('Age')
        plt.ylabel('Predicted probability (class=1)')
        plt.title('Predicted probability vs Age (test set)')
        age_file = Path.cwd() / "age_vs_prob_diagnostics.png"
        plt.savefig(age_file, bbox_inches='tight')
        print("Saved age vs prob plot to:", age_file)
        plt.show()

    # Print numeric summary
    print("ROC AUC:", f"{auc_val:.3f}")
    print(classification_report(y_test, preds))

    return model, scaler, X_test, y_test, probs

def compute_costs_from_wtp(wtp=50000, qaly_loss_per_missed=0.5, direct_cost_fn=5000, cost_fp=500):
    cost_fn = direct_cost_fn + (qaly_loss_per_missed * wtp)
    return float(cost_fp), float(cost_fn)

def threshold_sweep_nmb(df,
                        cost_fp=None,
                        cost_fn=None,
                        thresholds=None,
                        plot=True):
    X, y = prepare_features(df, binary=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    stratify=y, test_size=0.25,
    random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(solver='liblinear', max_iter=2000)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    rows = []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
        total_cost = fp * cost_fp + fn * cost_fn
        nmb = - total_cost
        rows.append({
            'threshold': thr,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'total_cost': total_cost,
            'nmb': nmb,
            'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) else 0.0
        })

    results = pd.DataFrame(rows)
    best_idx = results['nmb'].idxmax()
    best_row = results.loc[best_idx]

    print(f"Best threshold maximizing NMB: {best_row['threshold']:.3f}")
    print(f"At that threshold: TP={best_row['tp']} FP={best_row['fp']} FN={best_row['fn']} TN={best_row['tn']}")
    print(f"Total expected cost = £{best_row['total_cost']:.2f}  (NMB = £{best_row['nmb']:.2f})")

    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(results['threshold'], results['nmb'], label='NMB', color='tab:purple')
        plt.axvline(best_row['threshold'], color='grey', linestyle='--', label=f'best={best_row["threshold"]:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('NMB (monetary, £)')
        plt.title('NMB vs Threshold')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(results['threshold'], results['sensitivity'], label='Sensitivity', color='tab:blue')
        plt.plot(results['threshold'], results['specificity'], label='Specificity', color='tab:green')
        plt.axvline(best_row['threshold'], color='grey', linestyle='--', label=f'best={best_row["threshold"]:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Sensitivity and Specificity vs Threshold')
        plt.legend()
        plt.show()

    return results, float(best_row['threshold']), model, scaler

if __name__ == "__main__":
    import traceback
    try:
        df = load_and_clean_data()
        print("\nFirst 5 rows of cleaned data:")
        print(df.head())

        analyze_disease_burden(df)
        run_logistic_regression(df, binary=True)
        run_logistic_with_options(df, class_weight='balanced', resample_train=False, threshold=0.5)
        run_logistic_with_options(df, class_weight='balanced', resample_train=True, threshold=0.5)
        model, scaler, X_test, y_test, probs = plot_logistic_diagnostics(df, binary=True)

        out_df = pd.DataFrame({'prob': probs, 'true': y_test.values})
        out_df.to_csv(Path.cwd() / "logistic_test_probs.csv", index=False)
        print("Saved test probabilities to:", Path.cwd() / "logistic_test_probs.csv")

        cost_fp, cost_fn = compute_costs_from_wtp(
            wtp=50000,
            qaly_loss_per_missed=0.5,
            direct_cost_fn=5000,
            cost_fp=500
        )
        results_nmb, best_thr, model_nmb, scaler_nmb = threshold_sweep_nmb(
            df, cost_fp=cost_fp, cost_fn=cost_fn, plot=True
        )
        print("Recommended threshold (NMB):", best_thr)

        # best-row summary (TP/FP/FN/TN) -- use isclose and fallback to nearest if needed
        mask = np.isclose(results_nmb['threshold'].values, best_thr, atol=1e-8)
        if mask.any():
            best_row = results_nmb.loc[mask].iloc[0]
        else:
            nearest_idx = (np.abs(results_nmb['threshold'].values - best_thr)).argmin()
            best_row = results_nmb.iloc[nearest_idx]

        print("\nBest-threshold confusion counts (TP/FP/FN/TN):")
        print(f"TP={int(best_row['tp'])}  FP={int(best_row['fp'])}  FN={int(best_row['fn'])}  TN={int(best_row['tn'])}")

        # table of top candidate thresholds by NMB
        print("\nTop 10 thresholds by NMB:")
        cols = ['threshold','tp','fp','fn','tn','sensitivity','specificity','total_cost','nmb']
        print(results_nmb.sort_values('nmb', ascending=False)[cols].head(10).to_string(index=False))

        # save full sweep for inspection
        results_nmb.to_csv(Path.cwd() / "threshold_sweep_nmb_results.csv", index=False)
        print("Saved threshold sweep results to:", Path.cwd() / "threshold_sweep_nmb_results.csv")

    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR during run. Traceback written to error_log.txt")
        with open(Path.cwd() / "error_log.txt", "w", encoding="utf-8") as f:
            f.write(tb)
        print(tb)

