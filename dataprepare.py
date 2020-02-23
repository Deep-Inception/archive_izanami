#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '<style>\n    div#notebook-container    { width: 100%; }\n    div#menubar-container     { width: 65%; }\n    div#maintoolbar-container { width: 99%; }\n</style>\n\nimport importlib\nimportlib.reload(util)')


# In[2]:


import raceresults
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

class DataPrepare:
    def __init__(self):
        self.race_result_extracted = ""
        self.X = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
        self.y = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
        self.standard_scaler = preprocessing.StandardScaler()
        self.racer_id_encoder = OneHotEncoder(categories="auto", handle_unknown='ignore')
        self.racer_id_categories = ""
        self.place_encoder = OneHotEncoder(categories="auto", handle_unknown='ignore')
        self.place_categories = ""
    
    def data_prepare(self, race_results, is_train):
        race_results = race_results.copy()
        race_results_extracted_by_distance = race_results[race_results.DISTANCE ==1800]
        
        if is_train:
            self.race_result_extracted = self.extract_records(race_results_extracted_by_distance)
        else:
            self.race_result_extracted = race_results_extracted_by_distance
        
        _X_extracted = self.race_result_extracted.copy().drop(["RACE_TIME", "DISTANCE", "RACE_DATE"], axis=1).reset_index(drop=True)
        self.y = self.race_result_extracted.copy()["RACE_TIME"]
        
        #ワンホットエンコーディング
        _X_racer_id = _X_extracted["RACER_ID"]
        _X_place = _X_extracted["PLACE"]
        if is_train:
            _X_racer_id_encoded, self.racer_id_categories = _X_racer_id.factorize()
            _X_place_encoded, self.place_categories = _X_place.factorize()
            racer_id_pd, self.racer_id_encoder = self.one_hot_encode(_X_racer_id_encoded, self.racer_id_categories)
            place_pd, self.place_encoder = self.one_hot_encode(_X_place_encoded, self.place_categories)
        else:
            _X_racer_id_encoded = self.racer_id_categories.get_indexer(_X_racer_id)
            _X_place_encoded = self.place_categories.get_indexer(_X_place)
            racer_id_pd = self.one_hot_encode_for_test(_X_racer_id_encoded, self.racer_id_categories, self.racer_id_encoder)
            place_pd = self.one_hot_encode_for_test(_X_place_encoded, self.place_categories, self.place_encoder)
        
        _X_dropped = _X_extracted.drop(["RACER_ID", "PLACE"], axis=1)
        _X_1hot = pd.concat([_X_dropped, racer_id_pd, place_pd], axis=1)
        
        #スケーリング
        if is_train:
            exhibition_time_pd = self.standard_scale(_X_1hot, "EXHIBITION_TIME")
        else:
            exhibition_time_pd = self.standard_scale_for_test(_X_1hot, "EXHIBITION_TIME")
            
        _X_1hot_dropped = _X_1hot.copy().drop(["EXHIBITION_TIME"], axis=1)
        
        self.X = pd.concat([_X_1hot_dropped, exhibition_time_pd], axis=1)

    def extract_records(self, race_results):
        racer_id_count = race_results['RACER_ID'].value_counts()
        racer_id_count_extracted = racer_id_count[racer_id_count >= 30]
        race_results_extracted  = race_results[race_results['RACER_ID'].isin(racer_id_count_extracted.index.values.tolist())]
        return race_results_extracted
    
    def one_hot_encode(self, values, categories):
        encoder = OneHotEncoder(categories="auto", handle_unknown='ignore')
        one_hot_values = encoder.fit_transform(values.reshape(-1,1))
        return  pd.DataFrame(one_hot_values.toarray(), columns=categories), encoder
    
    def one_hot_encode_for_test(self, values, categories, encorder):
        encoder = encorder
        one_hot_values = encoder.transform(values.reshape(-1,1))
        return  pd.DataFrame(one_hot_values.toarray(), columns=categories)

    def standard_scale(self, records, column_name):
        records_num = records[[column_name]]
        records_num_scaled = self.standard_scaler.fit_transform(records_num)
        return pd.DataFrame(records_num_scaled, columns=[column_name])

    def standard_scale_for_test(self, records, column_name):
        records_num = records[[column_name]]
        records_num_scaled = self.standard_scaler.transform(records_num)
        return pd.DataFrame(records_num_scaled, columns=[column_name])
    
    def get_prepared_data(self):
        return self.X, self.y
    
    def get_extracted_data(self):
        return self.race_result_extracted


# In[3]:


if __name__ == "__main__":
    race_results = raceresults.RaceResults()
    data_prepare = DataPrepare()
    data_prepare.data_prepare(race_results.get_results_pd(), True)
    X, y = data_prepare.get_prepared_data()
    data_prepare.data_prepare(race_results.get_results_pd(), False)

