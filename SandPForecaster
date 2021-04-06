import pandas as pd
import yfinance as yf
import statsmodels.api as sm

data = yf.download("^GSPC", start='2001-01-02', end='2005-12-31')
df = data['Adj Close'].pct_change()*100
df = df.rename('Today')
df = df.reset_index()

for i in range(1, 6):
  df['Lag'+str(i)] = df['Today'].shift(i)
  df

df['Volume'] = data.Volume.shift(1).values/1000_000_000
df =df.dropna()
df['Direction'] = [1 if i>0 else 0 for i in df['Today']]
df = sm.add_constant(df)

X = df[['const', 'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
Y = df.Direction
model = sm.Logit(Y,X)
result = model.fit()
result.summary()

prediction = result.predict(X)


def confusion_matrix(act, pred):
  predtrans = ['Up' if i>.5 else "Down" for i in pred]
  actuals = ['Up' if i>0 else "Down" for i in act]
  confusion_matrix = pd.crosstab(pd.Series(actuals), pd.Series(predtrans), rownames=['Actual'], colnames=['Predicted'])
  return confusion_matrix
confusion_matrix(Y, prediction)

(155+504)/len(df)

x_train = df[df.Date.dt.year < 2005][['const', 'Lag2', 'Lag3','Lag4']]
y_train = df[df.Date.dt.year < 2005]['Direction']
x_test = df[df.Date.dt.year == 2005][['const', 'Lag2', 'Lag3', 'Lag4']]
y_test = df[df.Date.dt.year == 2005]['Direction']

model = sm.Logit(y_train, x_train)
result = model.fit()

prediction = result.predict(x_test)
confusion_matrix(y_test, prediction)

(25+128)/len(x_test)
