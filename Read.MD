## Code Explanation (Line by Line)

This code performs a forward feature evaluation experiment:

You test models with:

- 1 feature
- 2 features
- 3 features
- 4 features
- 5 features

And calculate:

- R² (train)
- Adjusted R² (train)
- R² (test)
- Adjusted R² (test)
- Test MSE

Then it plots how R² changes as features increase.

### ⭐ 1. Define the Base Model

```python 
base_sgd = SGDRegressor(
    loss="squared_error",
    learning_rate="invscaling",
    eta0=0.1,
    power_t=0.5,
    penalty=None,
    random_state=42,
    max_iter=2000,
    tol=1e-6,
)
```

This sets up one SGD linear regression model that will be reused.

- `loss="squared_error"` → model minimizes MSE
- `learning_rate="invscaling"` → learning rate shrinks over time
- `eta0=0.1` → initial learning rate
- `penalty=None` → no L1 or L2 regularization
- `max_iter=2000` → iterates many times
- `tol=1e-6` → stop when convergence is good
- `random_state=42` → reproducibility

### ⭐ 2. Loop through number of features (1 → 5)

```python 
for k in range(1, len(candidate_features) + 1):
    cols = candidate_features[:k] # Take the first element from the list until k-th
```

If our features list is:
```
["Previous Scores", "Hours Studied", "Sleep Hours", "Sample Questions", "Extracurricular Activities"]
```

Then:

- k=1 → use only ["Previous Scores"]
- k=2 → use ["Previous Scores", "Hours Studied"]
- k=3 → first 3 features
- … etc.

This simulates building the model step-by-step.

### ⭐ 3. Identify numeric and binary columns

```python
cur_numeric = [c for c in cols if c in numeric_cols]
cur_binary  = [c for c in cols if c in binary_cols]
```

Example:

- numeric features → scale with StandardScaler
- binary features → leave as-is

This ensures correct preprocessing

why we do this, and why we do NOT use numeric_cols and binary_cols directly.

Because in this loop, you are testing 1 feature → 2 features → 3 features → …
So you cannot use all numeric/binary features at once.

You need only the features chosen at this step.


### ⭐ 4. Create a ColumnTransformer

```python
cur_preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), cur_numeric),
        ("bin", "passthrough", cur_binary),
    ],
    remainder="drop",
)
```

This applies:

- StandardScaler() → on numeric columns
- "passthrough" → leave binary columns unchanged

This avoids scaling 0/1 data.

If my binary column is already converted to 0/1, why do I still need to put it inside ColumnTransformer with ‘passthrough’? Why not just ignore it?

Because ColumnTransformer ONLY outputs the columns you list inside it.
If you don’t list a column → it gets DROPPED.

Even if you already preprocessed the feature,
even if you don't need scaling or encoding…
you STILL need to include it in ColumnTransformer.
Otherwise it will disappear from the model input.

### ⭐ 5. Build a Pipeline: preprocessing + model

```python 
pipe = make_pipeline(cur_preproc, base_sgd)
```

Now training is:

1. scale data (data prep)
2. run SGDRegressor (model)

Automatically.

### ⭐ 6. Fit the model

```python 
pipe.fit(X_train[cols], y_train)
```

SGD will train using only the selected feature subset.

### ⭐ 7. Get predictions

```python 
y_tr = pipe.predict(X_train[cols])
y_te = pipe.predict(X_test[cols])
```

We evaluate model performance on:

- training data
- test data

### ⭐ 8. Compute R² and Adjusted R²

```python 
adj_tr, r2_tr = adjusted_r2(y_train, y_tr, p=len(cols))
adj_te, r2_te = adjusted_r2(y_test,  y_te, p=len(cols))
```

You pass:

- true y
- predicted y
- number of features

Your function returns:

- Adjusted R²
- R²

### ⭐ 9. Compute MSE

```python
mse_te = mean_squared_error(y_test, y_te)
```

### ⭐ 10. Store results

```python 
results.append({
    "k_features": k,
    "features": cols,
    "R2_train": r2_tr,
    "Adj_R2_train": adj_tr,
    "R2_test": r2_te,
    "Adj_R2_test": adj_te,
    "Test_MSE": mse_te
})
```

### ⭐ 11. Convert results to DataFrame

```python 
res_df = pd.DataFrame(results)
```

### ⭐ 12. Plot R² vs # features

```python 
plt.plot(res_df["k_features"], res_df["R2_test"])
plt.plot(res_df["k_features"], res_df["Adj_R2_test"])
```

This visualizes:

- Does R² increase?
- When does it flatten?
- When is adding more features pointless?

In your results, R² increased massively from k=1 → k=2,
but barely changed from k=2 → k=5.

This is why we choose 2 features only.