"""
log_parser.py
Turns a raw ICS log line into structured fields and loads the dataset files.

Example:
  2026-10-12T08:14:03Z src=10.0.9.77 device=PLC-01 event=unauthorized_write value=95 msg="..."
  -> {"timestamp": "2026-10-12T08:14:03Z", "src": "10.0.9.77", ..., "msg": "..."}

Missing fields are set to "-", so every model gets the same keys.
"""

import csv
import re

FIELDS = ["src", "dst", "device", "protocol", "event", "user", "count", "value", "msg"]

# key=value pairs; a value is either a quoted string or a run of non-space characters
PAIR_RE = re.compile(r'(\w+)=("([^"]*)"|\S+)')
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def parse_log(raw):
    """Parse one raw log line into a dict of fields."""
    fields = {k: "-" for k in FIELDS}
    ts = TIMESTAMP_RE.match(raw.strip())
    fields["timestamp"] = ts.group(0) if ts else "-"
    for key, whole, quoted in PAIR_RE.findall(raw):
        if key in fields:
            fields[key] = quoted if whole.startswith('"') else whole
    return fields


def to_model_text(fields, include_msg=True):
    """
    The text the Transformer reads. The timestamp is left out because it carries
    no information about severity. The free-text msg field can be excluded
    (used as a defense in the log-injection test).
    """
    keys = [k for k in FIELDS if include_msg or k != "msg"]
    return " ".join(f"{k}={fields[k]}" for k in keys)


def load_split(path):
    """Read a dataset CSV into a list of dicts with parsed fields added."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["fields"] = parse_log(r["raw"])
        r["actions"] = r["actions"].split()
    return rows


if __name__ == "__main__":
    example = ('2026-10-12T08:14:03Z src=10.0.9.77 dst=10.0.2.11 device=PLC-01 protocol=modbus '
               'event=unauthorized_write user=unknown count=1 value=95 msg="write to holding register"')
    f = parse_log(example)
    for k, v in f.items():
        print(f"{k:10s} {v}")
    print("\nModel text:", to_model_text(f))

