import statsmodels.api as sm

def sarimax_train_forecast(train_df, test_df, exog_cols=None, order=(1,0,1), seasonal_order=(1,1,1,24)):
    if exog_cols is None:
        exog_cols = [c for c in train_df.columns if c != 'y']
    model = sm.tsa.statespace.SARIMAX(train_df['y'], exog=train_df[exog_cols], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=len(test_df), exog=test_df[exog_cols]).predicted_mean
    return preds
