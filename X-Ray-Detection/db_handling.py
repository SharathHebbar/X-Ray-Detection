import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.environ.get('DB_NAME')
TABLE_NAME = os.environ.get('TABLE_NAME')

def create_db_if_not_exits():
    connection = sqlite3.connect(DB_NAME)
    cur = connection.cursor()
    return connection, cur

def create_table_if_not_exits(cur):
    connection, cur = create_db_if_not_exits()
    cur.execute("CREATE TABLE IF NOT EXISTS creds (Username TEXT NOT NULL, Password TEXT NOT NULL UNIQUE)")
    connection.commit()
    cur.close()
    connection.close()

def insert_data(username, password):
    connection, cur = create_db_if_not_exits()
    try:
        cur.execute("INSERT INTO creds (Username, Password) VALUES (?, ?)", (username, password))
    except Exception as e:
        return False
    connection.commit()
    cur.close()
    connection.close()
    return True

def delete_data(username, password):
    connection, cur = create_db_if_not_exits()
    cur.execute("DELETE FROM creds WHERE Username = (?) and Password = (?)", (username, password))
    connection.commit()
    cur.close()
    connection.close()


def update_password(username, password):
    connection, cur = create_db_if_not_exits()
    cur.execute("UPDATE creds SET Password = (?) WHERE Username = (?)", (password, username))
    connection.commit()
    cur.close()
    connection.close()    


def select_data(username=None, password=None):
    connection, cur = create_db_if_not_exits()
    if username is None and password is None:
        curr = cur.execute("SELECT * FROM creds ")
        connection.commit()
        return connection, cur, curr
    if username is None:
        curr = cur.execute("SELECT * FROM creds WHERE Password = (?)",(password))
        connection.commit()
        return connection, cur, curr
    if password is None:
        curr = cur.execute("SELECT * FROM creds WHERE Username = (?)",(username))
        connection.commit()
        return connection, cur, curr
    if password is not None and username is not None:
        curr = cur.execute("SELECT * FROM creds WHERE Username = (?) and Password = (?)",(username, password))
        connection.commit()
        return connection, cur, curr