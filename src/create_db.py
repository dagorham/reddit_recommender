__name__ = 'create_db'

import pandas as pd
import os


def create_db(conn):
    pwd = os.getcwd()
    print("Loading CSV.\n")
    temp_df = pd.read_csv(pwd + '/../db/comments.csv')
    print("Adding to DB.\n")
    temp_df.to_sql('comments', conn, if_exists='replace')
    print("Done.\n")