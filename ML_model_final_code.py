import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import PoissonRegressor
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from scipy.stats import poisson
import warnings

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# =====================================================================
# 1. FEATURE ENGINEERING FUNCTIONS
# =====================================================================
def add_all_features(df):
    epsilon = 1e-6
    
    # --- A. BASE INTERACTIONS ---
    df['home_matchup_strength'] = df['home_attack_strength'] * df['away_defense_strength']
    df['away_matchup_strength'] = df['away_attack_strength'] * df['home_defense_strength']
    df['strength_diff'] = df['home_matchup_strength'] - df['away_matchup_strength']
    df['home_attack_ratio'] = df['home_attack_strength'] / (df['away_defense_strength'] + epsilon)
    df['away_attack_ratio'] = df['away_attack_strength'] / (df['home_defense_strength'] + epsilon)
    
    # --- B. MATH FORMULAS ---
    df['elo_win_prob'] = 1 / (1 + 10 ** (-df['elo_diff'] / 400.0))
    df['log_strength_supremacy'] = np.log((df['home_matchup_strength'] + epsilon) / (df['away_matchup_strength'] + epsilon))
    gamma = 1.7
    df['home_pythagorean'] = (df['home_matchup_strength']**gamma) / (df['home_matchup_strength']**gamma + df['away_matchup_strength']**gamma + epsilon)
    
    # --- C. TIME-SERIES FORM & FATIGUE ---
    home_df = df[['date', 'home_team', 'home_score', 'away_score', 'home_attack_strength']].copy()
    home_df.columns = ['date', 'team', 'gf', 'ga', 'attack_strength']
    away_df = df[['date', 'away_team', 'away_score', 'home_score', 'away_attack_strength']].copy()
    away_df.columns = ['date', 'team', 'gf', 'ga', 'attack_strength']
    
    team_hist = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)
    team_hist['points'] = np.where(team_hist['gf'] > team_hist['ga'], 3, np.where(team_hist['gf'] == team_hist['ga'], 1, 0))
    team_hist['gd'] = team_hist['gf'] - team_hist['ga']
    team_hist['rest_days'] = team_hist.groupby('team')['date'].diff().dt.days
    
    # Identify Wins (1 = Win, 0 = Draw/Loss)
    team_hist['is_win'] = np.where(team_hist['points'] == 3, 1, 0)
    
    # Rolling Form Metrics (Shifted to prevent leakage)
    team_hist['ema_pts'] = team_hist.groupby('team')['points'].transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    team_hist['ema_gd'] = team_hist.groupby('team')['gd'].transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    team_hist['attack_momentum'] = team_hist.groupby('team')['attack_strength'].transform(lambda x: x - x.shift(5))
    team_hist['wins_last_5'] = team_hist.groupby('team')['is_win'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())

    # Merge Home
    df = df.merge(team_hist[['date', 'team', 'rest_days', 'ema_pts', 'ema_gd', 'attack_momentum', 'wins_last_5']], 
                  left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').rename(
                  columns={'rest_days': 'home_rest', 'ema_pts': 'home_ema_pts', 'ema_gd': 'home_ema_gd', 
                           'attack_momentum': 'home_attack_momentum', 'wins_last_5': 'home_wins_last_5'}).drop(columns=['team'])
    
    # Merge Away
    df = df.merge(team_hist[['date', 'team', 'rest_days', 'ema_pts', 'ema_gd', 'attack_momentum', 'wins_last_5']], 
                  left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').rename(
                  columns={'rest_days': 'away_rest', 'ema_pts': 'away_ema_pts', 'ema_gd': 'away_ema_gd', 
                           'attack_momentum': 'away_attack_momentum', 'wins_last_5': 'away_wins_last_5'}).drop(columns=['team'])
    
    # Advantages
    df['rest_advantage'] = df['home_rest'] - df['away_rest']
    df['form_advantage_gd'] = df['home_ema_gd'] - df['away_ema_gd']
    df['momentum_advantage'] = df['home_attack_momentum'] - df['away_attack_momentum']
    df['win_advantage_last_5'] = df['home_wins_last_5'] - df['away_wins_last_5']
    
    # Cleanup NAs
    df['home_rest'] = df['home_rest'].fillna(14)
    df['away_rest'] = df['away_rest'].fillna(14)
    df['rest_advantage'] = df['rest_advantage'].fillna(0)
    df.fillna(0, inplace=True) 
    
    return df

# =====================================================================
# 2. LOAD & PREPARE DATA
# =====================================================================
print("Loading and Engineering Data...")
df = pd.read_csv('montegoal_featureX.csv', parse_dates=['date']).dropna(subset=['home_score', 'away_score']).copy()
df = df.sort_values('date').reset_index(drop=True)
df = add_all_features(df)

# The Surgically Pruned 15 Features
BEST_FEATURES = [
    'elo_diff', 
    'is_home_advantage', 
    'home_attack_strength', 
    'away_defense_strength', 
    'away_attack_strength', 
    'home_defense_strength',
    'home_matchup_strength', 
    'away_matchup_strength', 
    'log_strength_supremacy', 
    'rest_advantage', 
    'home_ema_gd', 
    'away_ema_gd', 
    'home_wins_last_5', 
    'away_wins_last_5', 
    'momentum_advantage'
]

# Strict 85/15 Chronological Split
n = len(df)
val_end = int(n * 0.85)

train = df.iloc[:val_end].copy()
test = df.iloc[val_end:].copy()

X_train, y_train_h, y_train_a = train[BEST_FEATURES], train['home_score'], train['away_score']
X_test, y_test_h, y_test_a = test[BEST_FEATURES], test['home_score'], test['away_score']
actual_outcomes = np.sign(y_test_h.values - y_test_a.values)

# =====================================================================
# 3. SETUP MODELS & HYPERPARAMETERS
# =====================================================================
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), BEST_FEATURES)])

# 1. Poisson Regression (GLM)
poisson_reg_h = Pipeline([('prep', preprocessor), ('reg', PoissonRegressor(max_iter=1000, alpha=1.0))])
poisson_reg_a = Pipeline([('prep', preprocessor), ('reg', PoissonRegressor(max_iter=1000, alpha=1.0))])

# 2. XGBoost (Poisson)
xgb_params = {
    'objective': "count:poisson",
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 2.0,
    'reg_lambda': 2.0,
    'random_state': RANDOM_STATE
}
xgb_h = Pipeline([('prep', preprocessor), ('reg', XGBRegressor(**xgb_params))])
xgb_a = Pipeline([('prep', preprocessor), ('reg', XGBRegressor(**xgb_params))])

# 3. TabPFN Regressor
# Updated TabPFN Initialization
tabpfn_h = Pipeline([
    ('prep', preprocessor), 
    ('reg', TabPFNRegressor(random_state=RANDOM_STATE, ignore_pretraining_limits=True))
])

tabpfn_a = Pipeline([
    ('prep', preprocessor), 
    ('reg', TabPFNRegressor(random_state=RANDOM_STATE, ignore_pretraining_limits=True))
])

models = {
    'Poisson Regression': (poisson_reg_h, poisson_reg_a),
    'XGBoost (Poisson)': (xgb_h, xgb_a),
    'TabPFN': (tabpfn_h, tabpfn_a)
}

# =====================================================================
# 4. METRICS & EVALUATION FUNCTIONS
# =====================================================================
def get_dixon_coles_outcome(home_lambdas, away_lambdas, rho=-0.135, max_goals=10):
    preds = []
    for hl, al in zip(home_lambdas, away_lambdas):
        h_probs = poisson.pmf(np.arange(max_goals), hl)
        a_probs = poisson.pmf(np.arange(max_goals), al)
        p_home, p_draw, p_away = 0, 0, 0
        for i in range(max_goals):
            for j in range(max_goals):
                prob = h_probs[i] * a_probs[j]
                if i == 0 and j == 0: prob *= max(0, 1 - (hl * al * rho))
                elif i == 1 and j == 0: prob *= max(0, 1 + (hl * rho))
                elif i == 0 and j == 1: prob *= max(0, 1 + (al * rho))
                elif i == 1 and j == 1: prob *= max(0, 1 - rho)
                
                if i > j: p_home += prob
                elif i < j: p_away += prob
                else: p_draw += prob
        max_p = max(p_home, p_draw, p_away)
        if max_p == p_home: preds.append(1) 
        elif max_p == p_away: preds.append(-1)
        else: preds.append(0) 
    return np.array(preds)

def calc_both_within_one(actual_h, actual_a, pred_h, pred_a):
    h_rounded, a_rounded = np.round(pred_h), np.round(pred_a)
    h_match = np.abs(actual_h - h_rounded) <= 1
    a_match = np.abs(actual_a - a_rounded) <= 1
    return (h_match & a_match).mean() * 100

# =====================================================================
# 5. TRAIN & EVALUATE ALL MODELS
# =====================================================================
print(f"Training on {len(train)} matches... Testing on {len(test)} matches (Dates: {test['date'].min().date()} to {test['date'].max().date()})\n")

results = []

for model_name, (m_h, m_a) in models.items():
    # Fit Home and Away Models
    m_h.fit(X_train, y_train_h)
    m_a.fit(X_train, y_train_a)
    
    # Predict (Clip at 1e-6 to ensure positive values for Poisson Deviance)
    pred_h = np.clip(m_h.predict(X_test), 1e-6, None)
    pred_a = np.clip(m_a.predict(X_test), 1e-6, None)
    
    # Calculate Metrics
    acc = (actual_outcomes == get_dixon_coles_outcome(pred_h, pred_a)).mean() * 100
    within_1 = calc_both_within_one(y_test_h, y_test_a, pred_h, pred_a)
    mae_h = mean_absolute_error(y_test_h, pred_h)
    mae_a = mean_absolute_error(y_test_a, pred_a)
    dev_h = mean_poisson_deviance(y_test_h, pred_h)
    dev_a = mean_poisson_deviance(y_test_a, pred_a)
    
    results.append({
        'Model': model_name,
        '1X2 Accuracy %': acc,
        'Both ±1 Goal %': within_1,
        'Home MAE': mae_h,
        'Away MAE': mae_a,
        'Home Deviance': dev_h,
        'Away Deviance': dev_a
    })

# Output Summary Table
results_df = pd.DataFrame(results)

print("=" * 85)
print("FINAL MODEL COMPARISON RESULTS")
print("=" * 85)
print(f"{'Model':<20} | {'1X2 Acc %':<10} | {'±1 Goal %':<10} | {'Home MAE':<9} | {'Away MAE':<9} | {'Home Dev':<9} | {'Away Dev':<9}")
print("-" * 85)
for _, r in results_df.iterrows():
    print(f"{r['Model']:<20} | {r['1X2 Accuracy %']:<10.2f} | {r['Both ±1 Goal %']:<10.2f} | {r['Home MAE']:<9.4f} | {r['Away MAE']:<9.4f} | {r['Home Deviance']:<9.4f} | {r['Away Deviance']:<9.4f}")
print("=" * 85)

# =====================================================================
# 6. FEATURE IMPORTANCE (XGBoost)
# =====================================================================
xgb_importances = xgb_h.named_steps['reg'].feature_importances_

importance_df = pd.DataFrame({
    'Feature': BEST_FEATURES,
    'Importance': xgb_importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\nXGBOOST FEATURE IMPORTANCE RANKING (Home Goals)")
for _, row in importance_df.iterrows():
    print(f"{row['Feature']:<25} | {row['Importance']:.4f}")

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('XGBoost Feature Importance (Home Goals)', fontsize=14, pad=15)
plt.xlabel('Relative Importance (Gain)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()