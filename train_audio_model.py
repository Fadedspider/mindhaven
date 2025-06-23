import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import joblib
import optuna

# Load the data
data = pd.read_csv('ravdess_combined_features.csv')
print(f"Loaded {len(data)} samples")

# Check class distribution
print("Class distribution:")
print(data['mood'].value_counts())

# Check number of features
print(f"Number of features (excluding mood): {len(data.columns) - 1}")

# Handle missing values
data = data.dropna()

# Check if we have enough data
if len(data) < 10:
    print("Not enough data to train the model. Please collect more audio samples.")
    exit()

# Split into features and labels
X = data.drop('mood', axis=1)
y = data['mood']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("Label mapping:")
for label, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{label} -> {encoded}")
print(f"Encoded labels shape: {y_encoded.shape}")
print(f"Unique encoded labels: {np.unique(y_encoded)}")

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Compute sample weights for training data
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use in prediction
joblib.dump(scaler, 'scaler.pkl')

# Feature selection
selector = SelectKBest(score_func=f_classif, k=20)  # Select top 20 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")

# Save selected feature indices
joblib.dump(selector.get_support(), 'feature_selector.pkl')

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
print(f"Resampled training set shape: {X_train_resampled.shape}")

# Define Optuna objective for XGBoost
def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }
    model = XGBClassifier(eval_metric='mlogloss', random_state=42, **param)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    return np.mean(cv_scores)

# Optimize XGBoost hyperparameters
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=100)
best_xgb_params = study_xgb.best_params
print(f"Best XGBoost parameters: {best_xgb_params}")

# Define Optuna objective for LightGBM
def objective_lgbm(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0)
    }
    model = LGBMClassifier(random_state=42, force_col_wise=True, **param)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    return np.mean(cv_scores)

# Optimize LightGBM hyperparameters
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=100)
best_lgbm_params = study_lgbm.best_params
print(f"Best LightGBM parameters: {best_lgbm_params}")

# Define Optuna objective for MLP
def objective_mlp(trial):
    param = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (50, 50), (100, 50)]),
        'activation': trial.suggest_categorical('activation', ['tanh', 'relu']),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True),
        'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True)
    }
    model = MLPClassifier(random_state=42, max_iter=500, **param)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    return np.mean(cv_scores)

# Optimize MLP hyperparameters
study_mlp = optuna.create_study(direction='maximize')
study_mlp.optimize(objective_mlp, n_trials=100)
best_mlp_params = study_mlp.best_params
print(f"Best MLP parameters: {best_mlp_params}")

# Initialize models with the best parameters
models = {
    'LightGBM': LGBMClassifier(random_state=42, force_col_wise=True, **best_lgbm_params),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True, C=1.0, gamma='scale'),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42, **best_xgb_params),
    'MLP': MLPClassifier(random_state=42, max_iter=500, **best_mlp_params)
}

# Train and evaluate individual models
best_model_name = None
best_model = None
best_cv_score = 0
trained_models = {}
cv_scores_dict = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ['XGBoost', 'LightGBM']:
        model.fit(X_train_resampled, y_train_resampled, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_resampled))
    else:
        model.fit(X_train_resampled, y_train_resampled)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
    print(f"{name} Cross-Validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    y_pred = model.predict(X_test_selected)
    print(f"{name} Test Set Evaluation:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    mean_cv_score = np.mean(cv_scores)
    cv_scores_dict[name] = mean_cv_score
    trained_models[name] = model
    if mean_cv_score > best_cv_score:
        best_cv_score = mean_cv_score
        best_model_name = name
        best_model = model

# Weighted Voting Ensemble
weights = {name: score / sum(cv_scores_dict.values()) for name, score in cv_scores_dict.items()}
print(f"Ensemble weights: {weights}")
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in trained_models.items()],
    voting='soft',
    weights=[weights[name] for name in trained_models.keys()]
)
voting_clf.fit(X_train_resampled, y_train_resampled)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_clf, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
print(f"\nWeighted Voting Ensemble Cross-Validation Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
y_pred = voting_clf.predict(X_test_selected)
print("Weighted Voting Ensemble Test Set Evaluation:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Compare and save the best model
ensemble_cv_score = np.mean(cv_scores)
if ensemble_cv_score > best_cv_score:
    best_model_name = 'Weighted Voting Ensemble'
    best_model = voting_clf
    best_cv_score = ensemble_cv_score

# Save the best model
joblib.dump(best_model, 'audio_mood_classifier.pkl')
print(f"\nBest model ({best_model_name}) saved as audio_mood_classifier.pkl")

# Save the feature columns
feature_columns = selected_features
joblib.dump(feature_columns, 'feature_columns.pkl')
print("Feature columns saved as feature_columns.pkl")