# Changelog

## 1.0.0 (v1 line, in progress)

Breaking. The v1 line changes the routing API; `main` (0.1.0) is unaffected and
still reproduces results produced before it.

### Fixed — runoff was routed through its own reach even when it should not be

`include_index_diag=False` (now `route_src_reach=False`) was meant to describe runoff that is already at the
**outlet** of its own reach, so that reach must not be traversed: the path kernel
`K(d, s)` should convolve the IRFs over `succ(s)..d`.

It did not. The diagonal was correctly replaced by the identity
(`LTIRouter`, `y = x + Kx`), but every off-diagonal path kernel still convolved
over `s..d` — the same kernel as `include_index_diag=True`. `closure_sub` moved
the *starting* destination but always began the path sum at the source node, so
each path carried one extra copy of the source reach's own IRF.

Consequences:

- travel time from a node to any downstream gauge no longer includes that node's
  own reach;
- **headwater reaches now carry no gradient** with `route_src_reach=False` — their parameters
  are genuinely unused, which is the intended meaning;
- routed discharge changes, so **checkpoints trained before this release are not
  comparable**. Check out `main` / `v0.1.0` to reproduce older results.

`route_src_reach=True` (the default) is unaffected and was always correct.

Why the test suite did not catch it: `tests/test_backward.py` compared the Triton
kernels against a reference that encoded the same formula, so both agreed on the
wrong answer. `tests/test_semantics.py` is new and tests routing behaviour
directly — path composition decoded from `pure_lag` arrival times, the headwater
gradient being exactly zero, and the cross-mode identity
`route(x, src=True) == route(irf_i * x_i, src=False)`.

### Changed — `include_index_diag` replaced by `route_src_reach`

`include_index_diag` named an implementation detail — a diagonal in a sparse
kernel — rather than the modelling choice it encodes, which is whether a node's
runoff is routed through that node's own reach. On `RivTree` and
`RivTreeCluster`:

```python
route_src_reach = True   # default, unchanged behaviour; runoff enters at the
                         # reach head and traverses that reach   (was True)
route_src_reach = False  # runoff is already at the reach outlet (was False)
```

One name, one meaning, all the way down: `init_pre_indices`,
`downstream_path_stats`, `closure_sub`, `transitive_closure` and `aggregate_irf`
all take `route_src_reach` rather than a differently-named internal flag.

### Changed — closure internals

- `closure_sub(head, tail, edges, path_cumsum, ...)` now takes two prefix
  tensors: the source end reads the inclusive prefix `P` when `route_src_reach`
  and the exclusive prefix `Q` otherwise, while the destination end always reads
  `Q`. Both kernels lost their `edges[dest]` lookup and outlet branch.
- New `ops.transitive_closure.downstream_prefixes(irf, edges)` returning `(P, Q)`.
- `RivTree` / `RivTreeCluster` / `LTIRouter` docstrings now state what the mode
  means physically.

### Known limitation

`RivTreeCluster` transfers inject an upstream cluster's routed discharge as
*runoff* at the receiving node, which only reconstructs the unsplit network when
that runoff traverses the receiving reach — i.e. only with `runoff_entry="head"`.
Every construction site in `io.py` already uses head mode. Addressed next in the
v1 line.
