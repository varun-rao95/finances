from argparse import ArgumentParser
import pandas as pd
from db import get_conn, insert_transactions, Transaction
import os
from constants import MONARCH_ACCOUNT_NAME_MAPPING  # Import the mapping


def main(dry_run=True):
    # Load the Monarch CSV file
    monarch_df = pd.read_csv(
        os.path.expanduser(
            "~/Downloads/transactions-192069774303065161-52d7a35a-18f1-4fca-b8e0-906d5dd23e67.csv"
        )
    )

    latest_dates = {}

    with get_conn() as conn:
        cursor = conn.cursor()

        # Retrieve the latest transaction date for each account
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
            print(
                "Dry-run mode enabled. Records dumped to your downloads folder for review:"
            )
            filtered_df.to_csv(os.path.expanduser("~/Downloads/filtered_monarch_txns.csv"))
        else:
            # Insert the filtered data into the SQLite database
            transactions = [
                Transaction(
                    Date=row["Date"],
                    Description=row["Merchant"],
                    OriginalDescription=row["Original Statement"],
                    Amount=abs(row["Amount"]),
                    TransactionType="credit" if row["Amount"] > 0 else "debit",
                    Category=row["Category"],
                    AccountName=row["Account"],
                    Labels=row.get("Tags", ""),
                    Notes=row.get("Notes", "")
                )
                for _, row in filtered_df.iterrows()
            ]
            insert_transactions(transactions)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Write Monarch transactions to SQLite database."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing to the database.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run mode enabled. No changes will be made to the database.")
        main(dry_run=True)
    else:
        print("Changes will be made to the database.")
        main(dry_run=False)
