"""Complete training workflow, identical to notebook code cells."""
import matplotlib
matplotlib.use("Agg")  # Save charts without opening blocking windows.
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Run the notebook in the project root, or run src/train.py from any directory.
ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd()
DATA = ROOT / 'data'
for folder in ['results', 'images', 'models']:
    (ROOT / folder).mkdir(exist_ok=True)
SEED = 42
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
print('scikit-learn:', sklearn.__version__)


FEATURES = """duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent hot num_failed_logins logged_in num_compromised root_shell su_attempted num_root num_file_creations num_shells num_access_files num_outbound_cmds is_host_login is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate dst_host_rerror_rate dst_host_srv_rerror_rate""".split()
CATEGORIES = ['protocol_type', 'service', 'flag']
NUMERIC = [c for c in FEATURES if c not in CATEGORIES]
COLUMNS = FEATURES + ['attack_name', 'difficulty']

def load_data(filename):
    frame = pd.read_csv(DATA / filename, header=None)
    if frame.shape[1] != 43:
        raise ValueError(f'{filename}: expected 43 columns, got {frame.shape[1]}')
    frame.columns = COLUMNS
    if frame['attack_name'].isna().any():
        raise ValueError('Missing target labels cannot be imputed.')
    frame['attack_name'] = frame['attack_name'].str.strip().str.rstrip('.')
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    frame[NUMERIC] = frame[NUMERIC].replace([np.inf, -np.inf], np.nan)
    return frame

train = load_data('KDDTrain+_20Percent.txt')
test = load_data('KDDTest+.txt')
assert len(train) == 25192 and len(test) == 22544
# Diagnose repeated feature records before training. Retain benchmark test rows.
audit = {
    'train_rows_raw': len(train), 'test_rows': len(test),
    'train_missing_feature_values': int(train[FEATURES].isna().sum().sum()),
    'test_missing_feature_values': int(test[FEATURES].isna().sum().sum()),
    'train_duplicate_feature_rows': int(train.duplicated(FEATURES).sum()),
    'test_duplicate_feature_rows': int(test.duplicated(FEATURES).sum()),
}
train = train.drop_duplicates(subset=FEATURES, keep='first').reset_index(drop=True)
train_hash = pd.util.hash_pandas_object(train[FEATURES], index=False)
test_hash = pd.util.hash_pandas_object(test[FEATURES], index=False)
audit['test_rows_matching_train_features'] = int(test_hash.isin(set(train_hash)).sum())
# Do not train on any feature record that appears in the supplied test set.
train = train.loc[~train_hash.isin(set(test_hash))].reset_index(drop=True)
audit['train_rows_used'] = len(train)
X_train, X_test = train[FEATURES], test[FEATURES]
y_train = (train['attack_name'] != 'normal').astype(int)
y_test = (test['attack_name'] != 'normal').astype(int)
assert list(X_train.columns) == FEATURES
assert 'difficulty' not in X_train and 'attack_name' not in X_train
print(json.dumps(audit, indent=2))
print(X_train.head(3).to_string(index=False))


counts = pd.DataFrame({'Training': y_train.value_counts(), 'Test': y_test.value_counts()}).fillna(0).astype(int)
counts.index = counts.index.map({0: 'Normal', 1: 'Attack'})
print(counts.to_string())
print('\nNumeric summary:')
print(X_train[['duration', 'src_bytes', 'dst_bytes', 'count']].describe().round(2).to_string())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
counts.plot.bar(ax=axes[0], color=['#245675', '#79a6b9'], rot=0)
axes[0].set(title='Normal and attack connections', ylabel='Connections', xlabel='')
train.loc[train.attack_name != 'normal', 'attack_name'].value_counts().head(6).sort_values().plot.barh(ax=axes[1], color='#245675')
axes[1].set(title='Most common training attacks', xlabel='Connections', ylabel='')
fig.tight_layout()
fig.savefig(ROOT / 'images/eda.png', dpi=170)
plt.show()


numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
category_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, NUMERIC),
    ('categorical', category_pipeline, CATEGORIES),
])
model = Pipeline([
    ('preprocess', preprocessor),
    ('forest', RandomForestClassifier(n_estimators=100, max_depth=20,
        min_samples_leaf=2, class_weight='balanced', random_state=SEED, n_jobs=-1)),
])
start = time.perf_counter()
model.fit(X_train, y_train)
training_seconds = time.perf_counter() - start
print(f'Trained on {len(X_train):,} connections in {training_seconds:.2f} seconds.')
print('Encoded input columns:', len(model['preprocess'].get_feature_names_out()))


start = time.perf_counter()
y_pred = model.predict(X_test)
prediction_seconds = time.perf_counter() - start
report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['Normal', 'Attack'], output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = [int(v) for v in cm.ravel()]
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'attack_precision': report['Attack']['precision'],
    'attack_recall': report['Attack']['recall'],
    'attack_f1': report['Attack']['f1-score'],
    'false_positive_rate': fp / (fp + tn),
    'miss_rate': fn / (fn + tp),
    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    'training_seconds': training_seconds, 'batch_prediction_seconds': prediction_seconds,
    'train_rows': len(X_train), 'test_rows': len(X_test),
    'always_normal_accuracy': float((y_test == 0).mean()),
    'sklearn_version': sklearn.__version__, 'seed': SEED,
}
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Normal', 'Attack'], digits=4, zero_division=0))
print(f'False alarms: {fp:,}; missed attacks: {fn:,}')
print(f'Always-normal baseline accuracy: {metrics["always_normal_accuracy"]:.2%}')
fig, ax = plt.subplots(figsize=(5.3, 3.6))
ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack']).plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
ax.set_title('Unseen test set confusion matrix')
fig.tight_layout()
fig.savefig(ROOT / 'images/confusion_matrix.png', dpi=170)
plt.show()
(ROOT / 'results/metrics.json').write_text(json.dumps(metrics, indent=2))
(ROOT / 'results/data_audit.json').write_text(json.dumps(audit, indent=2))
pd.DataFrame(report).T.to_csv(ROOT / 'results/classification_report.csv')


analysis = pd.DataFrame({'attack_name': test.attack_name, 'actual_attack': y_test, 'predicted_attack': y_pred})
subtypes = analysis.loc[analysis.actual_attack == 1].groupby('attack_name').agg(support=('predicted_attack', 'size'), detected=('predicted_attack', 'sum'))
subtypes['missed'] = subtypes['support'] - subtypes['detected']
subtypes['detection_recall'] = subtypes['detected'] / subtypes['support']
subtypes = subtypes.sort_values('support', ascending=False)
subtypes.to_csv(ROOT / 'results/attack_type_recall.csv')
analysis.to_csv(ROOT / 'results/test_predictions.csv', index=False)
print(subtypes.to_string())
fig, ax = plt.subplots(figsize=(7.5, 3.4))
subtypes.head(8).sort_values('detection_recall')['detection_recall'].mul(100).plot.barh(ax=ax, color='#245675')
ax.set(xlabel='Attack detection recall (%)', ylabel='', xlim=(0, 105), title='Eight most common test attack types')
fig.tight_layout()
fig.savefig(ROOT / 'images/attack_recall.png', dpi=170)
plt.show()
print(f'The model misses {fn:,} attacks ({metrics["miss_rate"]:.2%}). It can support an analyst, but it is not sufficient as the only defense.')


joblib.dump(model, ROOT / 'models/ids_pipeline.joblib', compress=3)
X_test.head(10).to_csv(ROOT / 'data/demo_features.csv', index=False)
(ROOT / 'models/feature_columns.json').write_text(json.dumps(FEATURES, indent=2))
restored = joblib.load(ROOT / 'models/ids_pipeline.joblib')
assert np.array_equal(restored.predict(X_test.head(10)), y_pred[:10])
# Only load joblib models from trusted sources.
replay_rows = []
print('SIMULATED STREAM: recorded features, no live packet capture')
for i in range(10):
    start = time.perf_counter()
    scores = restored.predict_proba(X_test.iloc[[i]])[0]
    prediction = int(restored.classes_[np.argmax(scores)])
    elapsed = (time.perf_counter() - start) * 1000
    result = {'record': i + 1, 'prediction': 'ATTACK' if prediction else 'NORMAL',
              'attack_score': float(scores[list(restored.classes_).index(1)]),
              'inference_ms': elapsed}
    replay_rows.append(result)
    print(f'{i+1:02d}  {result["prediction"]:6}  attack score={result["attack_score"]:.3f}  inference={elapsed:.1f} ms')
pd.DataFrame(replay_rows).to_csv(ROOT / 'results/replay_output.csv', index=False)
print('All results, charts and the saved pipeline are ready.')
