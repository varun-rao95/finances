import pandas as pd
import sqlite3
import json
import os
from constants import MONARCH_ACCOUNT_NAME_MAPPING

dry_run = True  # Set to False to perform actual database insertion/updates

# Load the Monarch CSV file
monarch_df = pd.read_csv(os.path.expanduser("~/Downloads/all_monarch_txns.csv"))

# Map Monarch account names to match database naming
monarch_df["Account"] = monarch_df["Account"].map(MONARCH_ACCOUNT_NAME_MAPPING)

# Connect to the SQLite database
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "mint_transactions.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: Ensure an archive table exists with the same structure as transactions
if not dry_run:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions_archive AS SELECT * FROM transactions WHERE 1=0;
    """
    )
    conn.commit()

# Step 2: Find overlapping Mint transactions from Sep 9, 2023, to current date
overlapping_mint_transactions = pd.read_sql_query(
    """
    SELECT * FROM transactions
    WHERE Date < '2023-09-09'
""",
    conn,
)

# Apply the account name mapping to the overlapping Mint transactions for consistent comparison
overlapping_mint_transactions["AccountName"] = overlapping_mint_transactions[
    "AccountName"
].map({v: k for k, v in MONARCH_ACCOUNT_NAME_MAPPING.items()})

# Initialize a dictionary to store the description mappings
description_mapping = {}

# Step 3: Reconcile transactions by matching on Date and Amount, and archive before updating
for index, row in monarch_df.iterrows():
    # Check for matches in overlapping Mint transactions
    match = overlapping_mint_transactions[
        (overlapping_mint_transactions["Date"] == row["Date"])
        & (overlapping_mint_transactions["Amount"] == row["Amount"])
        & (overlapping_mint_transactions["AccountName"] == row["Account"])
    ]

    # If a match is found, archive the transaction and then update the description
    if not match.empty:
        # Archive the transaction
        if dry_run:
            print(
                f"[DRY RUN] Would archive transaction on {row['Date']} for {row['Amount']} in account {row['Account']}"
            )
        else:
            cursor.execute(
                """
                INSERT INTO transactions_archive
                SELECT * FROM transactions
                WHERE Date = ? AND Amount = ? AND AccountName = ?
            """,
                (row["Date"], row["Amount"], row["Account"]),
            )

        # Store the original and updated descriptions in the mapping dictionary
        original_description = match.iloc[0]["Description"]
        description_mapping[original_description] = row["Merchant"]

        # Update the transaction description
        if dry_run:
            print(
                f"[DRY RUN] Would update description from '{original_description}' to '{row['Merchant']}' for transaction on {row['Date']} in account {row['Account']}"
            )
        else:
            cursor.execute(
                """
                UPDATE transactions
                SET Description = ?
                WHERE Date = ? AND Amount = ? AND AccountName = ?
            """,
                (row["Merchant"], row["Date"], abs(row["Amount"]), row["Account"]),
            )

# Step 4: Save the description mapping dictionary to the data/ directory
if dry_run:
    print(
        f"[DRY RUN] Would save description mapping dictionary to data/description_mapping.json"
    )
else:
    with open("data/description_mapping.json", "w") as f:
        json.dump(description_mapping, f, indent=4)
        print("Saved description mapping dictionary to data/description_mapping.json")

# Step 5: Apply description mapping to transactions before Sep 9, 2023
for original_desc, updated_desc in description_mapping.items():
    if dry_run:
        print(
            f"[DRY RUN] Would update historical description from '{original_desc}' to '{updated_desc}' for transactions before Sep 9, 2023"
        )
    else:
        cursor.execute(
            """
            UPDATE transactions
            SET Description = ?
            WHERE Description = ? AND Date < '2023-09-09'
        """,
            (updated_desc, original_desc),
        )

# Commit changes if not in dry-run mode
if not dry_run:
    conn.commit()

# Close the connection
conn.close()
