from array import array
import joblib
import json
import numpy as np 
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from Input import Input
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('pred_model')

path = 'books.csv'
df = pd.read_csv(path, error_bad_lines=False)

df2 = df.copy()

df2.loc[ (df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[ (df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[ (df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[ (df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[ (df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

rating_df = pd.get_dummies(df2['rating_between'])
language_df = pd.get_dummies(df2['language_code'])

features = pd.concat([rating_df, 
                      language_df, 
                      df2['average_rating'], 
                      df2['ratings_count']], axis=1)
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

dist, idlist = model.kneighbors(features)


def BookRecommender(book_name):
    book_list_name = []
    book_id = df2[df2['title'] == book_name].index
    
    book_id = book_id[0]

   
    
    for newid in idlist[book_id]:
        book_list_name.append(df2.loc[newid].title)
    return book_list_name



@app.get('/')
async def root():
    return {'message': 'Hello World'}



    # prediction = model.predict([array])
    # if prediction[0] == 1:
    #     return 'Real User'
    
    # elif prediction[0] == 0:
    #     return 'Fake User'
    # else:
    #     return''



@app.post('/predict')
def predict(data: Input):
    data = data.dict()
    name=data['name']
    booknames = BookRecommender(name)
    return (booknames) 


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
