# NBA rookies: Predicting a 5-year career

A classifier that estimates whether an NBA rookie will remain in the league for more than 5 years, based on rookie-season statistics, and a small REST API that serves it. Originally written in February 2023 as a take-home exercise for a data science role; rewritten and extended in October 2026.

> **Data:** the dataset is confidential and is not included in this repository.
> To run the code, place a CSV with the same structure in this folder as `nba_logreg.csv`.

## Problem

Investors want to identify rookies likely to last more than 5 years. Missing a future long-career player costs more than a false alarm, so the model is evaluated on **recall**, with **precision** always reported next to it.

## Data

One row per rookie, with: games and minutes played, points per game, field goals, three-pointers and free throws (attempted, made, percentage), offensice and defensive rebounds, assists, steals, blocks, turnovers, and the target `TARGET_5Yrs` (1 = career of 5+ years). About 62% of players are positives. Missing three-point percentages (players with no attempt) are filled with 0.

## Approach

1. **Exploration** (`exploration.ipynb`): class balance, correlations, PCA and feature importances.
2. **Model comparison:** SVC, Random Forest, Decision Tree and Gradient Boosting, on all features, PCA components and a decorrelated subset, with cross-validation. 
3. **Selection:** SVC on all features had the best recall with almost no gap between training and held-out accuracy. Gradient Boosting was close on recall but overfit more.
4. **Tuning:** Grid Search on a training split. The selection rule keeps the best recall among configurations whose precision clearly beats the base rate, because recall alone picks models that predict "yes" for everyone.
5. **Final model:** the best configuration, refitted on all the data.

## Results

| Metric | Value |
|---|---|
| Recall (5+ years class) | 0.86 |
| Precision (5+ years class) | 0.74 |
| Accuracy | 0.72 |

The model finds most long-career players, but half of the players who will not last are still flagged. With 335 test players, the estimates carry roughly 5 points of uncertainty. Rookie statistics only partly predict career length, so the output is best used to shortlist players, not as a final decision.

## Repository

| File | Purpose |
|---|---|
| `exploration.ipynb` | Exploration and model comparison |
| `train.py` | Tuning, evaluation, saves the final model |
| `integration.py` | Flask API serving the model |
| `requirements.txt` | Dependencies (tested with Python 3.9) |

## Run

```bash
pip install -r requirements.txt
python train.py        # creates model.pkl and model_columns.pkl
python integration.py  # starts the API on http://127.0.0.1:5000
```

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"GP": 60, "MIN": 20.5, "PTS": 8.1, "...": 0}'
```

Example response:

```json
{"prediction": "Player will remain in the NBA for more than 5 years", "confidence": 0.78}
```

`confidence` is the model's probability for the predicted class. The API
expects every feature, accepts `null` for missing values, and returns a
400 error naming any missing feature. Only load pickle files you trained yourself.

## Limitations and next steps

- The 0.5 threshold is the default: lowering it approaches "predict yes for everyone", raising it loses recall quickly. The right value depends on the invesors' cost of a wrong "yes" versus a missed player.
- A random split does not test performance on future draft class; splitting by draft year would. 
- Possible additions: error analysis, tests for the API.