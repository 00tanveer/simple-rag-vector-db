'''
Database connection utilities and helpers (e.g., connect, disconnect, run queries).
'''

import psycopg
from psycopg import sql
import os
from dotenv import load_dotenv

def db_feed_data_batch(data):
    print("Feeding data into database...")
    conn = get_conn()
    with conn.cursor() as cur:
        for fact in data:
            cur.execute('INSERT INTO cat_facts (fact) VALUES (%s) ON CONFLICT (fact) DO NOTHING', (fact.strip(),), prepare=False)
        conn.commit()
        db_close(conn)

def db_init():
    print("Initializing database...")
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS cat_facts (
            id SERIAL PRIMARY KEY,
            fact TEXT NOT NULL UNIQUE,
            embedding VECTOR(1024)
        )
        ''')
        conn.commit()
        db_close(conn)
def get_conn():
    load_dotenv()
    return psycopg.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"))

def db_close(conn):
    conn.close()