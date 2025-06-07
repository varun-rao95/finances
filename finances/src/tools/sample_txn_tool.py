# /tools/get_txns.py
"""
{
  "description": "Return transactions between start_date and end_date.",
  "allowed_use": "unlimited",          // or an int
  "params": {
    "start_date": "str  # YYYY-MM-DD",
    "end_date":   "str  # YYYY-MM-DD"
  },
  "output": [
    { "id": "int"   },
    { "date": "str" },
    { "amount": "float" }
  ],
  "error_codes": {
    "E42": "Date range too large"
  },
  "dependencies": [
    "sqlite-utils>=3.36.0"
  ]
}
"""
from dataclasses import dataclass
