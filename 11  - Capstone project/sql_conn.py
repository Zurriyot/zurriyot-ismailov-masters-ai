import sqlite3
import pandas as pd

# Database and table details
db_file = "campaigns.db"  # Change to your database file
table_name = "campaigns"  # Change to your table name

def get_campaigns():
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)

    # Read the table into a Pandas DataFrame
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)

    # Close the connection
    conn.close()
    return df

def add_campaign(query):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()


    # Insert data into the table
    cursor.execute(query)

    # Commit the changes and close the connection
    conn.commit()

    # Close the connection
    conn.close()
    return "success"
