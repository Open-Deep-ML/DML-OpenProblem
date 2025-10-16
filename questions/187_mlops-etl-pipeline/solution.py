from typing import List, Tuple


def run_etl(csv_text: str) -> List[Tuple[str, float]]:
	"""Reference ETL implementation.

	- Extract: parse CSV text, skip header, strip whitespace, ignore blanks
	- Transform: keep event_type == "purchase"; parse value as float; aggregate per user
	- Load: return sorted list of (user_id, total_value) by user_id asc
	"""
	lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
	if not lines:
		return []
	# header
	header = lines[0]
	rows = lines[1:]

	# indices from header (allow varying order and case)
	headers = [h.strip().lower() for h in header.split(",")]
	try:
		idx_user = headers.index("user_id")
		idx_event = headers.index("event_type")
		idx_value = headers.index("value")
	except ValueError:
		# header missing required columns
		return []

	aggregates: dict[str, float] = {}
	for row in rows:
		parts = [c.strip() for c in row.split(",")]
		if len(parts) <= max(idx_user, idx_event, idx_value):
			continue
		user_id = parts[idx_user]
		event_type = parts[idx_event].lower()
		if event_type != "purchase":
			continue
		try:
			value = float(parts[idx_value])
		except ValueError:
			continue
		aggregates[user_id] = aggregates.get(user_id, 0.0) + value

	return sorted(aggregates.items(), key=lambda kv: kv[0])
