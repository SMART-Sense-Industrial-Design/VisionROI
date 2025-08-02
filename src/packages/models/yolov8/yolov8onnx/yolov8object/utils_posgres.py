
def conn_posgres(counter):
    import psycopg2

    # Connect to the database
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="nisitsirimarnkit",
        user="postgres",
        password=""
    )

    # Create a cursor object
    cur = conn.cursor()

    # Check if the table exists
    cur.execute("SELECT EXISTS (SELECT relname FROM pg_class WHERE relname = 'defect');")

    # Fetch the result
    table_exists = cur.fetchone()[0]

    if not table_exists:
        # Create the table
        cur.execute("CREATE TABLE defect (id SERIAL PRIMARY KEY,timestamp TIMESTAMP NOT NULL DEFAULT (current_timestamp AT TIME ZONE 'Asia/bangkok'), detail VARCHAR(255), counter INTEGER)")

        # Insert data into the table
        cur.execute("INSERT INTO defect (detail, counter) VALUES (%s, %s)", ("leak", counter))

        # Commit the changes to the database
        conn.commit()
    else:
        print("Table already exists.")
         # Insert data into the table
        cur.execute("INSERT INTO defect (detail, counter) VALUES (%s, %s)", ("leak", counter))
        # Commit the changes to the database
        conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

# conn_posgres(4)