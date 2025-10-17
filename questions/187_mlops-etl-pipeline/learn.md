## Solution Explanation

This task mirrors a minimal MLOps ETL flow that prepares data for downstream modeling.

### ETL breakdown
- Extract: parse raw CSV text, ignore blanks, and split into header and rows.
- Transform:
	- Filter only relevant records (event_type == "purchase").
	- Cast `value` to float; discard invalid rows to maintain data quality.
	- Aggregate total purchase value per user to create compact features.
- Load: return a deterministic, sorted list of `(user_id, total_value)`.

### Why this design?
- Input sanitation prevents runtime errors and poor-quality features.
- Aggregation compresses event-level logs into user-level features commonly used in models.
- Sorting produces stable, testable outputs.

### Complexity
- For N rows, parsing and aggregation run in O(N); sorting unique users U costs O(U log U).

### Extensions
- Add schema validation and logging.
- Write outputs to files or databases.
- Schedule ETL runs and add monitoring for drift and freshness.
