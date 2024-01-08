# coding: utf-8
import csv
import sqlite3
from datetime import datetime

from PIL import Image
import pytesseract

def convert_date_format(date_str):
    # Parse the date from MM/dd/YYYY format
    dt = datetime.strptime(date_str, '%m/%d/%Y')
    # Format the date to YYYY-MM-dd format
    return dt.strftime('%Y-%m-%d')


def dump_mint_csv():
    # Path to your CSV file
    csv_file_path = '/Users/varunrao/Downloads/transactions.csv'
    # Open the CSV file and insert data into the database
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cursor.execute('''
                INSERT INTO transactions (Date, Description, OriginalDescription, Amount, TransactionType, Category, AccountName, Labels, Notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (convert_date_format(row['Date']), row['Description'], row['Original Description'], row['Amount'], row['Transaction Type'], row['Category'], row['Account Name'], row['Labels'], row['Notes']))


def dump_mission_lane_statement():
    # Path to your PDF file
    img_paths = [f'/Users/varunrao/Downloads/mission_lane_{a}_dec.png' for a in ("desc", "amount")]
    
    for img_path in img_paths:
        image = Image.open(img_path)

        # Perform OCR using pytesseract
        extracted_text = pytesseract.image_to_string(image, config='--psm 4')
        print(extracted_text)

    return extracted_text

if __name__ == "__main__":
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('mint_transactions.db')
    cursor = conn.cursor()

    # Create a table transactions table
    cursor.execute('''
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
    ''')

    # dump_mint_csv()
    dump_mission_lane_statement()

    # Commit changes and close the connection
    conn.commit()
    conn.close()


