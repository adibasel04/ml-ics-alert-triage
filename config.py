"""
config.py
Shared settings for the ML-Assisted ICS Alert Triage prototype.

Everything here describes a SIMULATED industrial site: device names, IP
addresses, safe operating ranges and the follow-up actions an analyst might
take. No real ICS data is used anywhere in this project.
"""

import os

# File locations (all relative to the project root)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

SEED = 42  # one seed everywhere so results are reproducible

# Simulated site
# Each device has an IP address and a safe range for its process value
# (for example a tank level or a temperature setpoint).
DEVICES = {
    "PLC-01": {"ip": "10.0.2.11", "safe_range": (20, 80)},
    "PLC-02": {"ip": "10.0.2.12", "safe_range": (30, 90)},
    "PLC-03": {"ip": "10.0.2.13", "safe_range": (10, 60)},
    "RTU-01": {"ip": "10.0.3.21", "safe_range": (0, 100)},
    "RTU-02": {"ip": "10.0.3.22", "safe_range": (40, 110)},
    "HMI-01": {"ip": "10.0.1.31", "safe_range": (0, 100)},
}

# Machines that normally talk to the devices
KNOWN_SOURCES = ["10.0.1.31", "10.0.1.40", "10.0.1.41"]   # HMI + engineering workstations
UNKNOWN_SOURCES = ["10.0.9.77", "192.168.50.4", "172.16.4.200"]

KNOWN_USERS = ["operator1", "operator2", "eng_admin"]
UNKNOWN_USERS = ["unknown", "guest", "svc_test"]

PROTOCOLS = ["modbus", "dnp3"]

# Labels
SEVERITIES = ["low", "medium", "high", "critical"]

# Recommended analyst actions for each (event, severity) pair.
# The models predict these action codes; triage.py turns them into sentences.
ACTIONS = {
    "monitor": "Keep monitoring; no action needed right now.",
    "verify_user": "Confirm with the account owner that this activity was expected.",
    "check_network": "Check the network path and device health for the cause.",
    "review_change": "Review the change against the approved change log.",
    "verify_write": "Verify the write command with the operations team.",
    "block_source": "Block the source IP at the firewall until it is investigated.",
    "isolate_device": "Isolate the affected device from the network if safe to do so.",
    "notify_engineer": "Notify the on-call control engineer immediately.",
    "restore_setpoint": "Restore the setpoint to a safe value and confirm the process is stable.",
}
