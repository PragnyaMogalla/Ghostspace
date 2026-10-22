# GhostSpace AI - Discovering Hidden Room-Usage Patterns

Smart-building ML project: estimate how many people are in a room from cheap sensors (temperature, light, sound, CO2, PIR motion),
find recurring usage states, and flag **ghost periods** - lights on while nobody is present.

## Run
```bash
pip install -r requirements.txt
# OPTIONAL but recommended: put the real UCI file at data/Occupancy_Estimation.csv
#   https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation  (download -> unzip)
python train.py     # preprocessing, models, evaluation -> models/
python app.py       # dashboard + API at http://localhost:5000
```
Without the CSV, `train.py` falls back to a synthetic dataset with the identical schema (flagged on the dashboard).
**Re-run `python train.py` after adding the real CSV - all numbers update automatically.**

## Course outcomes -> where they live
| Outcome | Implementation |
|---|---|
| 1 Preprocessing | `train.py: preprocess()` - timestamp parsing, de-duplication, median imputation, feature engineering, z-score scaling, SelectKBest |
| 2 Classification / regression | Logistic Regression, KNN (k tuned by CV), Gaussian NB; Linear, Ridge, KNN regressors |
| 3 Clustering | K-Means, Ward hierarchical, GMM; elbow, silhouette, Davies-Bouldin, BIC/AIC, ARI/NMI |
| 4 Evaluation | accuracy, macro precision/recall/F1, ROC-AUC (OvR + binary), 5-fold CV, time-split test, SGD log-loss convergence, MAE/RMSE/R2 |
| 5 End-to-end + deployment | Flask API (`/api/predict`, `/api/results`, `/api/sample`) + dashboard in `static/index.html` |

## Method notes (good for Q&A)
- **Ghost definition:** the dataset has no energy label. A "ghost" sample = mean light above a data-driven threshold (30% of the median light level when occupied) AND true count = 0. kWh is an estimate from a user-set lighting load.
- **Leakage:** consecutive 30-second readings are highly correlated, so the random split is optimistic. We also report a *time-split* score (train on first 75% of time, test on last 25%).
- **Imbalance:** the room is empty most of the time, so we report macro-averaged metrics, not just accuracy.
- **Clusters never see labels;** labels are used only afterwards (ARI/NMI) to judge them.
- **API inputs:** the UI sends 6 aggregates; `features.from_aggregates` expands them to the 16 raw sensors using a typical occupied reading as template.
- Hierarchical clustering is evaluated on a 4,000-row subsample (O(n^2) memory).

Link: https://ghostspace.onrender.com/
