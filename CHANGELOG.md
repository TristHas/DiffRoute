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

### Fixed — clustered graphs now route identically to the unsplit graph

Splitting a network into clusters is a scheduling device, so it must not change
any output value. That only held with `route_src_reach=True`. The transfer hands
an upstream cluster's *routed discharge* to the downstream cluster as if it were
runoff, which reconstructs the unsplit network only if the receiving row still
traverses exactly one reach — true when the source reach is routed, false when
it is not, since then a node's input is already past its own reach.

Resolved by handing the value to a **copy of the upstream breakpoint node**
placed in the downstream cluster, rather than to the downstream node itself.
Delivering it straight to `v` cannot work in general, because `v`'s input would
then be a mixture: local runoff that must not traverse `irf_v`, and transferred
discharge that must. The copy separates them.

`route_src_reach` is therefore resolved **per node**:

- `RivTree(..., transition_nodes=...)` marks copies. They are always False
  whatever the global setting, and emit no output of their own.
- `own_reach` is a per-node buffer; the closure kernel reads it instead of a
  compile-time flag, and `select_head` becomes `where(own_reach, P, Q)` — the
  uniform cases still alias `P` or `Q` with no extra allocation.
- `route_src_reach` also accepts a `{node: bool}` mapping.
- `RivTreeCluster.nodes` is the duplicate-free layout the router takes in and
  hands back; `internal_nodes` exposes the layout with copies.
- `define_schedule` returns `(clusters, node_transfer, transition_nodes)` and now
  segments with `add_edge=True`.

`tests/test_clusters.py` asserts clustered == unsplit, node for node, both ways.

Three robustness bugs surfaced by single-reach clusters, all fixed:

- `init_pre_indices` raised on an edge-less graph (`zip(*g.edges)`).
- `aggregate_irf` called a bare `.squeeze()` on the IRF table, which dropped the
  node dimension when `n == 1`. It now reshapes to `[n, -1]`.
- `LTIRouter` built a kernel even when no path carries anything (a single reach
  with `route_src_reach=False`), which crashed the conv. It now returns the
  residual alone.

### Added — route to a selected set of output reaches

`RivTree(..., output_reach=[...])`, or `set_output_reach(...)` at any time later,
restricts what the router returns. Paths that end at a reach nobody asked about
are dropped from the closure **and** from the convolution kernel, whose row
dimension becomes the number of selected reaches:

```
20000 reaches, 400 gauges
  paths    6,670,319 ->   147,202   ( 45.3x)
  blocks      39,313 ->     7,488   (  5.3x)
  kernel  (20000, 20000) -> (400, 20000)
  routing     32.57 ms ->     5.54 ms   (  5.9x)
```

Selection is **structural**, not a forward argument: it is resolved once against
the existing `nodes_idx`, on the host, so nothing has to be looked up per call
and no device-side index is needed. `set_output_reach(None)` restores full
routing.

What it deliberately does **not** touch: the graph, the node ordering, and the
runoff tensor the router expects, which stays one row per node in graph order.
`prefix_sum` and the IRF transform therefore still run over every node; narrowing
those is a separate problem.

- `downstream_path_stats` takes an optional `keep`, so the counts — and hence the
  write offsets — follow the selection.
- the closure kernel reads `out_row`: the destination's row in the output, or -1
  to drop it. One tensor is both the predicate and the remap, so the emitted
  kernel is `(n_out x n_in)` rather than `(n x n)` with holes.
- the block-sparse convolution already supported a non-square kernel on its
  default (triton) path — `conv_temp_1D.py` carries `C_out`/`C_in` and separate
  `n_in_blocks`/`n_out_blocks` in both directions. Only the non-default legacy
  torch fallback assumes square.

Not supported on `RivTreeCluster`: a cluster's boundary node has to stay in its
own cluster's output, since that value is what `node_transfer` hands downstream.
It raises rather than silently dropping a transfer source.

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
