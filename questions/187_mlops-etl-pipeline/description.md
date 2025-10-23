## Problem

Implement a simple ETL (Extract-Transform-Load) pipeline for model-ready data preparation.

Given a CSV-like string containing user events with columns: `user_id,event_type,value` (header included), write a function `run_etl(csv_text)` that:

1. Extracts rows from the raw CSV text.
2. Transforms data by:
	- Filtering only rows where `event_type == "purchase"`.
	- Converting `value` to float and dropping invalid rows.
	- Aggregating total purchase `value` per `user_id`.
3. Loads the transformed results by returning a list of `(user_id, total_value)` tuples sorted by `user_id` ascending.

Assume small inputs (no external libs), handle extra whitespace, and ignore blank lines.
