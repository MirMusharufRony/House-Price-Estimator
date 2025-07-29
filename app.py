#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
df=pd.read_csv("kc_house_data.csv")
print(df.head())
df.isnull().sum()
#%%
df.describe().transpose()
# %%
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(df['price'])
# %%
sns.countplot(df['bedrooms'])
# %%
df.corr(numeric_only=True)['price'].sort_values()
# %%
sns.scatterplot(x='price', y='sqft_living', data=df)
# %%
sns.scatterplot(x='price', y='grade', data=df)
# %%
sns.boxplot(x='bedrooms', y='price', data=df)
df.columns
# %%
import plotly.express as px

fig = px.scatter_mapbox(df,
                        lat="lat",
                        lon="long",
                        hover_name="id",
                        hover_data={"price": True, "sqft_living": True, "bedrooms": True},
                        color="price",
                        size="sqft_living",
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size_max=10,
                        zoom=9,
                        mapbox_style="open-street-map")  # <-- Google Maps-like style

fig.update_layout(title="King County House Locations (Google Maps-like Style)")
fig.show()
fig.write_html("house_map.html")
# %%
sns.scatterplot(x='price', y='long',data= df)

# %%
sns.scatterplot(x='price', y='lat', data=df)
# %%
sns.scatterplot(x='long', y='lat', data=df, hue='price')
# %%
df.sort_values('price', ascending=False).head(20)
# %%
len(df)
# %%
non_top_1_pec=df.sort_values('price', ascending=False).iloc[216:]
# %%
plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',data=non_top_1_pec,hue='price', edgecolor=None, palette='RdYlGn', alpha=0.2)
# %%
df.head()
# %%
sns.boxplot(x='waterfront', y='price',data=df)
# %%
df= df.drop('id',axis=1)
# %%
df.head()
# %%
df['date']=pd.to_datetime(df['date'])
# %%
df['date']
# %%
df['year']= df['date'].apply(lambda date: date.year)
# %%
df.head()
# %%
df['month']= df['date'].apply(lambda date: date.month)
# %%
df.head()
# %%
plt.figure(figsize=(12,8))
sns.boxplot(x='month', y='price', data=df)
# %%
df.groupby('month').mean()['price']
# %%
df.groupby('month').mean()['price'].plot()
# %%
df.groupby('year').mean()['price'].plot()
# %%
df=df.drop('date',axis=1)
# %%
df['zipcode'].value_counts()
# %%
df= df.drop('zipcode',axis=1)
# %%
df.head()
# %%
df['yr_renovated'].value_counts()
# %%
X= df.drop('price', axis=1).values
y= df['price'].values
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# %%
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
# %%
X_train
# %%
X_test= scaler.transform(X_test)
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# %%
X_train.shape
# %%
model= Sequential()
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# %%
model.fit(x=X_train,y= y_train, validation_data=(X_test,y_test),batch_size=128,epochs=400)
# %%
model.history.history
# %%
losses=pd.DataFrame(model.history.history)
# %%
losses.plot()
# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
# %%
pred= model.predict(X_test)
# %%
mean_absolute_error(y_test, pred)
# %%
np.sqrt(mean_squared_error(y_test,pred))
# %%
explained_variance_score(y_test,pred)
# %%
single_house=df.drop('price',axis=1).iloc[0]
# %%
single_house= scaler.transform(single_house.values.reshape)
# %%
model.predict(single_house)
# %%
df.head(1)
# %%
