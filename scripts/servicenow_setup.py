"""One-time ServiceNow setup: check the login and create the 9 assignment groups used by the demo.

Run from the project root:  .venv/bin/python scripts/servicenow_setup.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.servicenow import ServiceNowClient

client = ServiceNowClient.from_env()
print(f"Connected to {client.instance} as {client.ping()}")
for name, sys_id in client.ensure_groups().items():
    print(f"  group ready: {name:24s} {sys_id}")
print("ServiceNow is ready for the demo.")
