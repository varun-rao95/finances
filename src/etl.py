# coding: utf-8
import csv
import sqlite3
from datetime import datetime

from PIL import Image
import pytesseract


def convert_date_format(date_str, from_="intuit"):
    if from_ == "intuit":
        # Parse the date from MM/dd/YYYY format
        dt = datetime.strptime(date_str, "%m/%d/%Y")
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
                    convert_date_format(row["Date"]),
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
    # Path to your PDF file
    img_paths = [
        f"/Users/varunrao/Downloads/mission_lane_{a}_sep.png"
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

    # add transaction type and account name
    for row in rows:
        row.append("credit")
        row.append("Mission Lane")

    if dry_run:
        print(rows)
    else:
        for row in [tuple(x) for x in rows]:
            cursor.execute(
                """
                    INSERT INTO transactions (Date, OriginalDescription, Amount, TransactionType, AccountName)
                    VALUES (?, ?, ?, ?, ?)
                """,
                row,
            )

    return rows


if __name__ == "__main__":
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect("mint_transactions.db")
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
    dump_mission_lane_statement()

    # Commit changes and close the connection
    conn.commit()
    conn.close()
