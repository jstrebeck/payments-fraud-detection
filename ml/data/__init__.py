"""Synthetic transaction data (ADR-0003).

`ml.data.schema` and `ml.data.reference` have no heavy dependencies and are
safe to import from the API. `ml.data.generator` and `ml.data.io` need the
`data` extra (numpy, pyarrow).
"""
