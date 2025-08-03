from psycopg2 import pool

from . import config


connection_pool = pool.SimpleConnectionPool(
    1,
    10,
    host=config.DB_HOST,
    port=config.DB_PORT,
    database=config.DB_NAME,
    user=config.DB_USER,
    password=config.DB_PASSWORD,
)


def get_connection():
    return connection_pool.getconn()


def release_connection(conn):
    if conn:
        connection_pool.putconn(conn)

