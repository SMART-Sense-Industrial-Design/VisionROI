from psycopg2 import DatabaseError

from packages.db import get_connection, release_connection


def conn_posgres(counter: int) -> bool:
    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS (SELECT relname FROM pg_class WHERE relname = 'defect');"
                )
                table_exists = cur.fetchone()[0]
                if not table_exists:
                    cur.execute(
                        "CREATE TABLE defect (id SERIAL PRIMARY KEY,timestamp TIMESTAMP NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'Asia/bangkok'), detail VARCHAR(255), counter INTEGER)"
                    )
                cur.execute(
                    "INSERT INTO defect (detail, counter) VALUES (%s, %s)",
                    ("leak", counter),
                )
        return True
    except DatabaseError as exc:
        if conn:
            conn.rollback()
        print(f"Database error: {exc}")
        return False
    finally:
        release_connection(conn)

