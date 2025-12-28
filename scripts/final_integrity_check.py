
import os
import pandas as pd
import json
import datetime

base_dir = "data_cache/R_100/ticks"
files = sorted([f for f in os.listdir(base_dir) if f.endswith(".metadata.json")])

report = []
for f in files:
    with open(os.path.join(base_dir, f), 'r') as jf:
        meta = json.load(jf)
        report.append({
            "month": f.replace(".metadata.json", ""),
            "records": meta.get("record_count"),
            "start": datetime.datetime.fromtimestamp(meta.get("start_epoch")).strftime('%Y-%m-%d'),
            "end": datetime.datetime.fromtimestamp(meta.get("end_epoch")).strftime('%Y-%m-%d')
        })

df = pd.DataFrame(report)
print(df.to_string(index=False))

total_records = df['records'].sum()
print(f"\nTotal Ticks: {total_records:,}")
print(f"Data Coverage: {df['month'].iloc[0]} to {df['month'].iloc[-1]}")
