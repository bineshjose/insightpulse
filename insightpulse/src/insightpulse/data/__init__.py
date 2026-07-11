"""L1 — Data layer: connectors, ETL pipelines, and repositories.

Production data lives in Snowflake (PB scale, queried in place) with ADLS
for benchmark landing, PostgreSQL for app metadata, and Redis for the hot
working set; demo mode swaps all of it for the CSV panel in ``data/demo``
behind the same contracts.
"""
