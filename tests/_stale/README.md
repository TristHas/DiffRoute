# Stale tests — not collected, kept for their reference implementations

These four modules stopped being runnable some time before 2026 and were never
noticed, because none of them matches pytest's discovery pattern except
`test_helpers.py`, which contains helpers rather than tests. Their presence in
`tests/` made `pytest tests/` fail at *collection*, so the suite that does work
(`test_backward.py`) could not be run by the obvious command.

They are kept, not deleted: `sequential_models.py` holds naive sequential
routing models used as ground truth, and `tests.py` holds seven numerical
tests comparing the fast block-sparse path against them. That is exactly the
kind of thing worth reviving — deliberately, by someone who knows the intended
semantics.

## What is broken

| module | breakage |
|---|---|
| all four | began with `from imports import *`. No `imports.py` has ever existed in this repository — they were written against a notebook scratch file. **Fixed**: replaced with the four names actually used (`numpy as np`, `networkx as nx`, `torch`, `torch.nn.functional as F`). |
| all four | `from diffroute.utils import get_node_idxs`. `diffroute/utils.py` was deleted in `ccb22e2` ("Major refactoring"); the function now lives in `diffroute.structs.riv_graphs` and is re-exported from the package root. **Fixed**: `from diffroute import get_node_idxs`. |
| `gen_data.py` | `annotate_downstream_path_stats`, also deleted in `ccb22e2`, with no replacement. Its only caller was `generate_inputs()`, which nothing used. **Fixed**: dead function and import dropped. |
| `tests.py`, `gen_data.py` | `RoutingIRFAggregator` was renamed to `IRFAggregator` in `ccb22e2`. **Not fixed** — see below. |

## Why the rename was not just applied

The call sites have drifted further than the rename. They construct the
aggregator as `RoutingIRFAggregator(g, ...)` and `RoutingIRFAggregator(g,
nodes_idx, ...)`, but the class took neither `g` nor `nodes_idx` in its
constructor even *before* `ccb22e2` — the graph goes to `forward(g, params)`.
So these tests were written against a third, older API and have not run since
then.

Reviving them means deciding what each test intends, not renaming a symbol.
Applying the rename alone would produce tests that fail, or worse, pass without
testing what their names claim.

## To revive

1. Move the module back into `tests/`, renaming `tests.py` to something
   pytest collects (e.g. `test_routing_equivalence.py`).
2. Replace `RoutingIRFAggregator(g, ...)` with `IRFAggregator(...)` plus
   `agg(g, params)` at each call site, checking each test's intent against
   `diffroute/agg/kernel_aggregator.py`.
3. Confirm each test fails when the fast path is deliberately broken — an
   equivalence test that cannot fail is worse than no test.
