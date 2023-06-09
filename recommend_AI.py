from urllib.request import urlopen
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def recommend(shoe):
    url = "https://spring-store-api.herokuapp.com/api/products"
    response = urlopen(url)
    data_json = json.loads(response.read())
    # print(data_json)
    # data_file = data_json['content']
    data_file = data_json
    data_file = list(filter(lambda dictionary: dictionary['status'] == "1",data_file))
    df = pd.json_normalize(data_file, max_level=1)
    # print(df)
    cv = CountVectorizer(max_features=5000,stop_words='english')
    # vector.shape
    vector = cv.fit_transform(df['description']).toarray()
    similarity = cosine_similarity(vector)
    # similarity

    index = df[df['name'] == shoe].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    s = ""
    for i in distances[1:6]:
        s = s + str(df.iloc[i[0]].id) + ","
        # print (df.iloc[i[0]].id)
    return s
# print(recommend("Adidas NMD"))