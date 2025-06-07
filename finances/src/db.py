import sqlite3
import os
from dataclasses import dataclass
from typing import Iterable


# === Dataclasses for tables ===
@dataclass
class Portfolio:
    id: int | None
    ticker: str
    buy: bool
    num_shares: int
    amount: float
    date: str


@dataclass
class Transaction:
    Date: str
    Description: str
    OriginalDescription: str
    Amount: float
    TransactionType: str
    Category: str
    AccountName: str
    Labels: str | None
    Notes: str | None


# === Path + Connection setup ===
def get_conn():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "mint_transactions.db")
    cx = sqlite3.connect(db_path)
    cx.row_factory = sqlite3.Row
    return cx


# === Example: Insert + Fetch for Portfolio ===
def insert_portfolio(p: Portfolio) -> int:
    with get_conn() as cx:
        cur = cx.execute(
            """INSERT INTO portfolio (ticker, buy, num_shares, amount, date)
               VALUES (?, ?, ?, ?, ?)""",
            (p.ticker, p.buy, p.num_shares, p.amount, p.date),
        )
        return cur.lastrowid


def all_portfolios() -> Iterable[Portfolio]:
    with get_conn() as cx:
        for row in cx.execute("SELECT * FROM portfolio"):
            yield Portfolio(**row)


# === Example: Fetch + Insert for Transactions ===
def all_transactions() -> Iterable[Transaction]:
    with get_conn() as cx:
        for row in cx.execute("SELECT * FROM transactions"):
            yield Transaction(**row)


def insert_transaction(txn: Transaction) -> None:
    with get_conn() as cx:
        cx.execute(
            """
            INSERT INTO transactions (
                Date, Description, OriginalDescription, Amount, TransactionType,
                Category, AccountName, Labels, Notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                txn.Date,
                txn.Description,
                txn.OriginalDescription,
                txn.Amount,
                txn.TransactionType,
                txn.Category,
                txn.AccountName,
                txn.Labels,
                txn.Notes,
            ),
        )


def insert_transactions(txns: Iterable[Transaction]) -> None:
    with get_conn() as cx:
        cx.executemany(
            """
            INSERT INTO transactions (
                Date, Description, OriginalDescription, Amount, TransactionType,
                Category, AccountName, Labels, Notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    t.Date,
                    t.Description,
                    t.OriginalDescription,
                    t.Amount,
                    t.TransactionType,
                    t.Category,
                    t.AccountName,
                    t.Labels,
                    t.Notes,
                )
                for t in txns
            ],
        )
