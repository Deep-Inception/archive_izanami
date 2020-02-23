#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '<style>\n    div#notebook-container    { width: 100%; }\n    div#menubar-container     { width: 65%; }\n    div#maintoolbar-container { width: 99%; }\n</style>')


# In[2]:


import collections
import datetime
import glob
import os
import re
import sqlite3
import time
import urllib

import lhafile
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

COLUMN_LIST = ['RACE_TIME', 'RACER_ID', 'PLACE', 'DISTANCE', 'COURSE', 'EXHIBITION_TIME', 'RACE_DATE']

class RaceResults:
    def __init__(self):
        self.baseuri = "http://www1.mbrace.or.jp/od2/K/%s/k%s.lzh" # http://www1.mbrace.or.jp/od2/K/201612/k161201.lzh
        self.results = [] # List of (RACE_TIME, RACER_ID, PLACE, COURSE, EXHIBITION_TIME)

    def download(self, start, end):
        period = pd.date_range(start, end)

        for date in period:
            # Get file from the website
            dirname = date.strftime("%Y%m")
            lzhname = date.strftime("%y%m%d")
            uri = self.baseuri % (dirname, lzhname)
            savename = "./data/results/lzh/%s.lzh" % lzhname
            if not os.path.exists(savename):
                print("Send request to", uri)
                urllib.request.urlretrieve(uri, savename)
                time.sleep(3)

            unpackedpath = "./data/results/K%s.TXT" % lzhname
            unpackedname = os.path.basename(unpackedpath)
            if not os.path.exists(unpackedpath):
                print("Unpacking", savename)
                f = lhafile.Lhafile(savename)
                data = f.read(unpackedname)
                datastr = data.decode(encoding='shift-jis')
                fileobj = open(unpackedpath, "w")
                fileobj.write(datastr)
                fileobj.close()

    def load_and_regist(self):
        all_results = []
        for filename in glob.glob("./data/results/K1*.TXT"):
            race_date_str = filename.replace("./data/results/K", "20").replace(".TXT","")
            tdatetime = datetime.datetime.strptime(race_date_str, '%Y%m%d')
            race_date = datetime.date(tdatetime.year, tdatetime.month, tdatetime.day)
            with open(filename, "r") as f:
                get_place = False
                get_racer_info = False
                count = 1
                race_time = 0
                for line in f:
                    line_replaced =  line.replace("\u3000", "")
                    if line_replaced.find('BGN') > -1:
                        get_place = True
                    elif get_place:
                        _place_index = line_replaced.find('［成績］')
                        place = line_replaced[0:_place_index]
                        get_place = False
                    elif re.search(r'H\d{4}m', line_replaced, flags=0)  is not None:
                        distance = re.search(r'\d{4}', line_replaced, flags=0).group()
                    elif line_replaced.startswith("----"):
                        get_racer_info = True
                    elif get_racer_info and count < 7:
                        elems = line_replaced.split()
                        if elems[0] not in ['01','02','03','04','05','06']:
                            count += 1
                            continue
                        if elems[9] != '.':
                            _race_time = elems[9].split('.')
                            race_time = float(_race_time[0])*60 + float(_race_time[1]) + float(_race_time[2])/10
                        else:
                            race_time = race_time + 1
                        racer_id = elems[2]
                        course = float(elems[1])
                        exhibition_time = float(elems[6])
                        if race_date.month == 12:
                            all_results.append((race_time, racer_id, place, distance, course, exhibition_time, race_date))
                        all_results.append((race_time, racer_id, place, distance, course, exhibition_time, race_date))
                        count += 1
                    elif count == 7:
                        count = 1
                        get_racer_info = False
                        
        self.results = all_results
        regist(self.results)
                                                            
    def get_results_pd(self):
        return pd.DataFrame(get_results(), columns=COLUMN_LIST)


# In[3]:


column_list_type = "RACE_TIME REAL, RACER_ID TEXT, PLACE TEXT, DISTANCE INTEGER, COURSE INTEGER, EXHIBITION_TIME REAL, RACE_DATE TEXT"
column_list = "RACE_TIME, RACER_ID, PLACE, DISTANCE, COURSE, EXHIBITION_TIME, RACE_DATE"
dbname = "raceresults.db"

def regist(results):
    conn=sqlite3.connect(dbname)
    cursor = conn.cursor()

    drop_table = "DROP TABLE IF EXISTS raceresults"
    cursor.execute(drop_table)
    create_table = "CREATE TABLE raceresults (" + column_list_type + ")"
    cursor.execute(create_table)

    sql = "INSERT INTO raceresults (" + column_list + ") Values (?,?,?,?,?,?,?)"
    for item in results:
        cursor.execute(sql, item)
    conn.commit()
    conn.close()

def get_results():
    conn=sqlite3.connect(dbname)
    cursor = conn.cursor()
    get_records = "SELECT * FROM raceresults"
    cursor.execute(get_records)
    results = cursor.fetchall()
    conn.close()
    return results


# In[4]:


if __name__ == "__main__":
    r = RaceResults()
    r.download("2019-10-01","2019-12-31")
    r.load_and_regist()
    a = r.get_results_pd()


# In[ ]:




