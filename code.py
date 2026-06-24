import pandas as pd
data={'BedRoom':[2,1,3] , 'Area_sq_feet':[736,276,1156], 'Years_old':[20,20,20], 'Location':['city','village','city'] ,'Estimated_price':[4000000,1700000,9000000]}
df=pd.DataFrame(data)
print(df)
print("\n"+"="*80+"\n")
drop_first=True
df_encoded=pd.get_dummies(df,columns=['Location'],dtype=int)
print("Encoded dataframe:")
x=df_encoded[['BedRoom','Area_sq_feet','Years_old','Location_city','Location_village']]
y=df_encoded['Estimated_price']
print(x)
print(y)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
print("Model Training complete")
new_house=[[2,900,20,1,0]]
predicted_price=model.predict(new_house)
print(f'The predicted price is :${predicted_price[0]:,.2f}')

