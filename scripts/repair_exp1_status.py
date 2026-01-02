import json
from pathlib import Path

STATUS_FILE = "results/raw/exp1_status.json"
PENDING_FILE = "results/raw/exp1_pending_runs.json"

def main():
    if not Path(STATUS_FILE).exists():
        print(f"Status file not found: {STATUS_FILE}")
        return
    
    if not Path(PENDING_FILE).exists():
        print(f"Pending file not found: {PENDING_FILE}")
        return

    with open(STATUS_FILE, "r") as f:
        status_data = json.load(f)

    with open(PENDING_FILE, "r") as f:
        pending_data = json.load(f)
        
    pending_keys = set(pending_data.get("pending_runs", []))
    
    if not pending_keys:
        print("No pending keys found.")
        return

    print(f"Status file currently lists {status_data['completed_runs']} completed runs.")
    print(f"Found {len(pending_keys)} pending runs to remove from completed status.")

    completed_keys = set(status_data["completed_keys"])
    
    # Remove pending keys from completed_keys
    new_completed_keys = completed_keys - pending_keys
    removed_count = len(completed_keys) - len(new_completed_keys)
    
    status_data["completed_keys"] = sorted(list(new_completed_keys))
    status_data["completed_runs"] = len(new_completed_keys)
    
    print(f"Removed {removed_count} keys from status.")
    print(f"New completed count: {status_data['completed_runs']}")

    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=2)
        
    print("Status file updated.")

if __name__ == "__main__":
    main()
