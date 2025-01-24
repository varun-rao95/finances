import pandas as pd
import sqlite3
import os
from constants import MONARCH_ACCOUNT_NAME_MAPPING  # Import the mapping

# Set dry_run mode
dry_run = False  # Set to False to perform actual insertion

# Load the Monarch CSV file
monarch_df = pd.read_csv(os.path.expanduser("~/Downloads/all_monarch_txns.csv"))

# Establish a connection to the SQLite database
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "mint_transactions.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Retrieve the latest transaction date for each account
latest_dates = {}
for row in cursor.execute(
    "SELECT AccountName, MAX(Date) as LatestDate FROM transactions GROUP BY AccountName"
):
    account_name, latest_date = row
    latest_dates[account_name] = latest_date

# Apply the account name mapping and filter the DataFrame to include only new transactions
monarch_df["Account"] = monarch_df["Account"].map(MONARCH_ACCOUNT_NAME_MAPPING)
filtered_df = monarch_df[
    monarch_df.apply(
        lambda x: x["Date"] > latest_dates.get(x["Account"], "0000-00-00"), axis=1
    )
]

# Dry-run: display transactions that would be inserted
if dry_run:
    print("Dry-run mode enabled. Records dumped to your downloads folder for review:")
    filtered_df.to_csv(os.path.expanduser("~/Downloads/filtered_monarch_txns.csv"))
else:
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
                abs(row["Amount"]),
                "credit" if row["Amount"] > 0 else "debit",
                row["Category"],
                row["Account"],
                row.get("Tags", ""),
                row.get("Notes", ""),
            ),
        )

    # Commit changes only if not in dry-run mode
    conn.commit()

# Close the connection
conn.close()
