import csv
import sqlite3
from datetime import datetime
import os

from PIL import Image
import pytesseract


def convert_date_format(date_str, from_="intuit"):
    if from_ == "intuit":
        # Parse the date from MM/dd/YYYY format
        dt = datetime.strptime(date_str, "%m/%d/%Y")
    elif from_ == "sofi":
        # Parse the date from YYYY-MM-dd format
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    else:
        # Parse the date from MM/dd format
        year = "2023"
        dt = datetime.strptime(date_str + f"/{year}", "%m/%d/%Y")

    # Format the date to YYYY-MM-dd format
    return dt.strftime("%Y-%m-%d")


def dump_mint_csv():
    # Path to your CSV file
    csv_file_path = "/Users/varunrao/Downloads/transactions.csv"
    # Open the CSV file and insert data into the database
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cursor.execute(
                """
                INSERT INTO transactions (Date, Description, OriginalDescription, Amount, TransactionType, Category, AccountName, Labels, Notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    convert_date_format(row["Date"], from_="intuit"),
                    row["Description"],
                    row["Original Description"],
                    row["Amount"],
                    row["Transaction Type"],
                    row["Category"],
                    row["Account Name"],
                    row["Labels"],
                    row["Notes"],
                ),
            )


def dump_mission_lane_statement(dry_run=False):
    # Path to your images
    # screenshot each column in the mission lane statement
    # date, desc, amount
    img_paths = [
        os.path.expanduser(f"~/Downloads/mission_lane_{a}_jan.png")
        for a in ("date", "desc", "amount")
    ]

    total_amounts_parsed, total = [], 0
    for img_path in img_paths:
        image = Image.open(img_path)

        # Perform OCR using pytesseract
        extracted_text = pytesseract.image_to_string(image, config="--psm 4")
        vals = [txt.strip() for txt in extracted_text.splitlines() if txt]

        if "date" in img_path:  # assume date is first image we're parsing
            len_txns = len(vals)
            rows = [None] * len_txns

            for idx, val in enumerate(vals):
                rows[idx] = [convert_date_format(val, from_="missionlane")]
        else:
            if (
                len(vals) != len_txns + 1
            ):  # here we assume a total record for desc + amount
                raise ValueError(
                    "Mismatch in OCR for uploaded image {}".format(img_path)
                )

            for row, val in zip(rows, vals[:-1]):
                row.append(val)
                if "amount" in img_path:
                    total += float(val.replace(",", ""))

            total_amounts_parsed.append(vals[-1].lstrip("$").replace(",", "").strip())

    if float(total_amounts_parsed[-1]) != round(total, 2):
        raise ValueError(
            f"May not have parsed all the rows - parsed total {total_amounts_parsed[-1]} vs total {total}"
        )

    # Add transaction type and account name to each row
    for row in rows:
        row.append("credit")
        row.append("Mission Lane")

    if dry_run:
        print(rows)
        for row in [list(x) for x in rows]:
            if "2023-12" not in row[0]:
                row[0] = row[0].replace("2023", "2024")
                print(row)
    else:
        for row in [list(x) for x in rows]:
            # Delete the exact record if it already exists, using Date, OriginalDescription, and Amount
            # cursor.execute(
            #     """
            #     DELETE FROM transactions
            #     WHERE Date = ? AND OriginalDescription = ? AND Amount = ? AND TransactionType = ? AND AccountName = ?
            #     """,
            #     row,  # row now includes Date, OriginalDescription, Amount, TransactionType, and AccountName
            # )
            if "2023-12" not in row[0]:
                row[0] = row[0].replace("2023", "2024")
                print(row)
            # Insert the corrected row
            cursor.execute(
                """
                INSERT INTO transactions (Date, OriginalDescription, Amount, TransactionType, AccountName)
                VALUES (?, ?, ?, ?, ?)
                """,
                row,
            )

    return rows


def dump_sofi_csv():
    # Path to your CSV file
    csv_file_paths = (
        ("SOFI Checking", "/Users/varunrao/Downloads/SOFI_checking.csv"),
        ("SOFI Saving", "/Users/varunrao/Downloads/SOFI_savings.csv"),
    )
    # Open the CSV file and insert data into the database

    for x in csv_file_paths:
        account_name, csv_file_path = x
        with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                amnt = float(row["Amount"])
                if float(row["Amount"]) < 0:
                    ttype = "debit"
                    amnt = amnt * -1
                else:
                    ttype = "credit"
                cursor.execute(
                    """
                    INSERT INTO transactions (Date, Description, Amount, TransactionType, Category, AccountName)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        convert_date_format(row["Date"], from_="sofi"),
                        row["Description"],
                        amnt,
                        ttype,
                        row["Type"],
                        account_name,
                    ),
                )


def dump_apple_card_transactions():
    # Path to your CSV file (modify this to point to the correct location)
    csv_file_path = "/Users/varunrao/Downloads/apple_card_transactions_{month}.csv"
    # Open the CSV file and insert data into the database
    for month in [
        "2024-09",
        "2024-10",
        ...,
    ]:  # modify this to include the desired months
        with open(
            csv_file_path.format(month=month), newline="", encoding="utf-8"
        ) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Transform the data to match your existing database schema
                date = convert_date_format(row["Transaction Date"], from_="apple")
                description = row["Description"]
                amount = float(row["Amount (USD)"])
                transaction_type = "debit" if amount < 0 else "credit"
                merchant = row["Merchant"]
                category = row["Category"]
                account_name = "Apple Card"
                # Load the data into the database
                cursor.execute(
                    """
                    INSERT INTO transactions (Date, Description, Amount, TransactionType, Merchant, Category, AccountName)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        date,
                        description,
                        amount,
                        transaction_type,
                        merchant,
                        category,
                        account_name,
                    ),
                )


if __name__ == "__main__":
    print("FUKCYOU MINT DEPRECATED")

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect("hullo.db")
    cursor = conn.cursor()

    # Create a table transactions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            Date TEXT,
            Description TEXT,
            OriginalDescription TEXT,
            Amount REAL,
            TransactionType TEXT,
            Category TEXT,
            AccountName TEXT,
            Labels TEXT,
            Notes TEXT
        )
    """
    )

    # dump_mint_csv()
    dump_mission_lane_statement(dry_run=False)
    # dump_sofi_csv()

    # Commit changes and close the connection
    conn.commit()
    conn.close()
