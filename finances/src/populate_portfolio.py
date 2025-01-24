import csv
import sqlite3
import os
from datetime import datetime
import argparse
from unittest import skip

# Define date boundaries
START_SKIP = datetime.strptime("2025-03-31", "%Y-%m-%d")
END_SKIP = datetime.strptime("2025-09-29", "%Y-%m-%d")

DB_NAME = "mint_transactions.db"  # Adjust if needed


def parse_date_mmddyyyy(date_str):
    """
    Convert 'MM/DD/YYYY' => datetime, or return None if empty/invalid.
    Example: '12/31/2021' => datetime(2021, 12, 31).
    """
    if not date_str.strip():
        return None
    try:
        return datetime.strptime(date_str.strip(), "%m/%d/%Y")
    except ValueError:
        return None


def read_csv_skip_footer(path, skip_top=0, skip_bottom=14):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    # Remove top empty lines if needed
    if skip_top > 0:
        lines = lines[skip_top:]

    # Remove last `skip_bottom` lines (the disclaimers)
    if skip_bottom > 0 and len(lines) > skip_bottom:
        lines = lines[:-skip_bottom]

    reader = csv.DictReader(lines, delimiter=",", quotechar='"', skipinitialspace=True)
    return reader


def main(csv_path, dry_run=False, skip_rows=0, skip_bottom=14):
    """
    Reads the CSV at `csv_path`, inserts into `transactions`
    (unless Run Date is in [2022-03-31..2023-09-29])
    and inserts into `portfolio` only if 'Settlement Date' is present.

    If `dry_run` is True, print what would happen but do not commit.
    """
    RUN_DATE = "Activity Date"
    ACTION = "Description"
    ORIGINAL_DESC = "Description"
    SYMBOL = "Instrument"
    AMOUNT = "Amount"
    SETTLEMENT_DATE = "Settle Date"
    QUANTITY = "Quantity"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, DB_NAME)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    reader = read_csv_skip_footer(csv_path, skip_top=skip_rows, skip_bottom=skip_bottom)
    for row in reader:
        # 1) Parse Run Date
        run_date_str = row[RUN_DATE].strip()  # e.g. "12/31/2021"
        run_date = parse_date_mmddyyyy(run_date_str)
        if not run_date:
            # If run date is invalid or empty, skip
            continue

        # 2) Check if we skip inserting to transactions
        skip_transactions = START_SKIP <= run_date <= END_SKIP

        # 3) Prepare insertion into transactions if not skipping
        action = row[ACTION].strip()  # e.g. "YOU BOUGHT PERIODIC INVESTMENT ..."
        symbol = row[SYMBOL].strip()  # e.g. "FXAIX"
        original_desc = row[ORIGINAL_DESC].strip()  # CSV "Description" column

        # Parse "Amount ($)" as a float
        try:
            amount = float(
                row[AMOUNT].lstrip("(").rstrip(")").lstrip("$").replace(",", "")
            )
        except (ValueError, TypeError):
            amount = 0.0

        # Decide transaction type
        if "DIVIDEND" in action.upper():
            txn_type = "credit"
        else:
            txn_type = "debit"

        category = "Investments"
        account_name = "Individual"  # or adjust as needed

        # We'll store the date in YYYY-MM-DD format for the database
        date_for_db = run_date.strftime("%Y-%m-%d")

        if not skip_transactions:
            # If not skipping, either insert or print (dry-run)
            if dry_run:
                print(
                    f"[DRY RUN] Would insert into transactions: Date={date_for_db}, Desc='{action}', "
                    f"OrigDesc='{original_desc}', Amount={amount}, Type={txn_type}, "
                    f"Category={category}, Account={account_name}"
                )
            else:
                # Insert into transactions table
                cursor.execute(
                    """
                    INSERT INTO transactions
                    (Date, Description, OriginalDescription, Amount, TransactionType, Category, AccountName)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date_for_db,
                        action,
                        original_desc,
                        amount,
                        txn_type,
                        category,
                        account_name,
                    ),
                )

        # 4) Insert into `portfolio` only if Settlement Date is present
        settlement_str = row.get(SETTLEMENT_DATE, "").strip()
        settlement_date = parse_date_mmddyyyy(settlement_str)
        if settlement_date and row.get("Trans Code", "") in ["Buy", "Sell"]:
            # We have a valid settlement date => that means we put it in portfolio
            try:
                quantity = float(row[QUANTITY])
            except (ValueError, TypeError):
                quantity = 0.0

            # If "Action" indicates a buy
            # e.g. if "BOUGHT", "REINVESTMENT", or "DIVIDEND" => buy=True
            if row.get("Trans Code", ""):
                is_buy = "Buy" in row.get("Trans Code")
            else:
                action_upper = action.upper()
                is_buy = any(
                    word in action_upper
                    for word in ["BOUGHT", "REINVESTMENT", "DIVIDEND"]
                )

            ticker = symbol
            portfolio_amount = amount
            date_portfolio = settlement_date.strftime("%Y-%m-%d")

            if dry_run:
                print(
                    f"[DRY RUN] Would insert into portfolio: "
                    f"Ticker={ticker}, Buy={is_buy}, Shares={quantity}, Amount={portfolio_amount}, "
                    f"Date={date_portfolio}"
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO portfolio (ticker, buy, num_shares, amount, date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ticker, is_buy, quantity, portfolio_amount, date_portfolio),
                )

    # 5) Commit at the end if not dry_run
    if not dry_run:
        conn.commit()
        print("Committed all inserts.")
    else:
        print("DRY RUN complete. No changes committed.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a CSV into transactions & portfolio with date-based skips."
    )
    parser.add_argument("csv_path", help="Path to the CSV file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without inserting data.",
    )
    parser.add_argument(
        "--skip-rows", type=int, default=0, help="Number of rows to skip."
    )
    args = parser.parse_args()

    main(args.csv_path, dry_run=args.dry_run, skip_rows=args.skip_rows, skip_bottom=2)
