import pandas as pd
import sqlite3

# Load the Monarch CSV file
monarch_df = pd.read_csv("path_to_monarch_csv.csv")

# Establish a connection to the SQLite database
conn = sqlite3.connect("path_to_your_database.db")
cursor = conn.cursor()

# Retrieve the latest transaction date for each account
latest_dates = {}
for row in cursor.execute(
    "SELECT AccountName, MAX(Date) as LatestDate FROM transactions GROUP BY AccountName"
):
    account_name, latest_date = row
    latest_dates[account_name] = latest_date

# Filter the DataFrame to include only new transactions
filtered_df = monarch_df[
    monarch_df.apply(
        lambda x: x["Date"] > latest_dates.get(x["Account"], "0000-00-00"), axis=1
    )
]

# Insert the filtered data into the SQLite database
for index, row in filtered_df.iterrows():
    cursor.execute(
        """
    INSERT INTO transactions (Date, Description, OriginalDescription, Amount, TransactionType, 
                              Category, AccountName, Labels, Notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            row["Date"],
            row["Merchant"],
            row["Original Statement"],
            row["Amount"],
            "credit" if row["Amount"] > 0 else "debit",
            row["Category"],
            row["Account"],
            row.get("Tags", ""),
            row.get("Notes", ""),
        ),
    )

# Commit changes and close the connection
conn.commit()
conn.close()
