import pandas as pd
import sqlite3
import os
from constants import MONARCH_ACCOUNT_NAME_MAPPING  # Import the mapping

# Load the Monarch CSV file
monarch_df = pd.read_csv(os.path.expanduser("~/Downloads/all_monarch_txns.csv"))

# Establish a connection to the SQLite database
conn = sqlite3.connect("mint_transactions.db")
cursor = conn.cursor()

# Set dry_run mode
dry_run = True  # Set to False to perform actual insertion

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
    print("Dry-run mode enabled. The following records would be inserted:")
    print(
        filtered_df[
            [
                "Date",
                "Merchant",
                "Original Statement",
                "Amount",
                "Category",
                "Account",
                "Tags",
                "Notes",
            ]
        ]
    )
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
                row["Amount"],
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
