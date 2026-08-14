# Changelog

## 1.0.0 (v1 line, in progress)

### Fixed — int32 address overflow in the closure and block-sparse conv Triton kernels

Triton wraps int32 address arithmetic silently rather than trapping. Two
products exceed 2^31 at hourly resolution (768 taps) that never did at daily
(32 taps):

- the conv kernels' weight-block offset, `nzb * (K * block_size**2)`, past
  ~10,922 nonzero blocks — reachable by an ordinary scattered output
  selection (a gauge set, an arbitrary batch), not just a pathological one.
  Confirmed via a direct Triton wrap probe and by cross-checking a wrapping
  batch (18,584 blocks) against the identical reaches requested in a row
  order that stays under the threshold (5,469 blocks): pre-fix, the two
  disagreed almost everywhere (>99% of entries, not just at the edges) —
  this was silently producing wrong finite output as often as it crashed,
  depending on where the wrapped (negative) offset happened to land.
- the closure kernels' `row * n_feat`, once a build's path count times
  feature count exceeds 2^31 (a full-output flat closure at hourly
  resolution on a network of tens of thousands of reaches).

Fix: widen the first large factor of each offset product to int64 before
multiplying. Address arithmetic only — no value, dtype, or floating-point
computation order changes, so every daily-resolution kernel shape (which
never approaches either threshold) is unaffected bit for bit.

### Added — frequency-domain convolution for large tap counts

Fixing the overflow makes any block count safe, but not cheap:
`BlockSparseKernel` pads the (dest, src) path set out to whole
`block_size x block_size` tiles, which wastes space badly once the tap
count K is large and the output selection is scattered rather than
graph-local (2-4% fill is typical) — a few hundred MB of real kernel data
can pad out to tens of GiB. `BlockSparseCausalConv` now dispatches to a new
`ops.conv.freq_conv_coo` (FFT overlap-add directly on the COO kernel, no
block quantization at all) once the projected block storage crosses a
threshold (default 6 GiB), invisibly to the caller — `LTIRouter.forward`
no longer calls `to_block_sparse` at all, it just hands the aggregated
kernel to the conv module. The threshold is calibrated so every
daily-resolution (K~32) kernel stays on the unchanged Triton path
regardless of fill, by orders of magnitude of headroom.

`freq_conv_coo`'s ops (rfft/index_select/complex-mul/index_add/irfft) are
ordinary differentiable torch calls — gradients w.r.t. both the input and
the kernel values come from autograd directly, no hand-written backward
kernel to keep in sync with the forward.

Effect on a real hourly, ~1,600-gauge-output routing forward+backward:
35.3 GiB peak (int64 fix alone) -> 4.8 GiB (with the frequency dispatch). A
4,096-output arbitrary batch that still OOM'd after the int64 fix alone
(77.9 GiB forward, no room left for backward's same-sized buffer) now peaks
at 18.4 GiB forward+backward combined.

### Fixed — `RivTreeCluster.params` was a property, so in-place edits to it were silently discarded

It was `torch.cat([g.params for g in self.gs])` — recomputed fresh from the
sub-trees on *every* access. Routing reads `gs.params` once per forward and
slices it per cluster, so that was fine for params built directly from a
`param_df`, but an in-place edit after construction (the standard way to
switch a graph's temporal resolution, e.g. `g.params[:, 1] *= 24` for k
days -> hours) landed on nothing: the property's next read discarded it and
rebuilt from the untouched sub-tree buffers. A clustered graph routed at
hourly resolution this way silently used unscaled (daily-timescale)
parameters — headwaters barely affected, error accumulating downstream
with drainage area, exactly the observed signature.

`params` is now a real buffer, concatenated once at construction, same as
`RivTree`'s own `params` always were.

### Added — `set_output_reach` on `RivTreeCluster`

Previously refused (`NotImplementedError`) — see the "Not supported on
`RivTreeCluster`" note under the output-reach entry below, which this
supersedes.

A cluster boundary's transfer source has to stay in *its* cluster's local
output regardless of what the caller asked for globally, since that row is
what `node_transfer` hands the downstream cluster as input. Per cluster,
the local output is now (requested nodes that live in it) union (its own
transfer sources whose downstream cluster is itself needed) — propagated
backward through the cluster dependency graph in one reverse pass, since
clusters are already given in forward-topological order. A cluster with
nothing needed in it, directly or transitively, is skipped entirely: it
never routes and never enters the transfer bucket.

With `output_reach=None` this reduces to exactly the pre-existing
computation (`out_gather` *is* `keep_pos`, `out_ranges` *is* `node_ranges`)
on its own unchanged, most performance-sensitive code path; the new
bookkeeping (including remapping `src_transfer`'s local index, which is a
position in a cluster's own discharge tensor and so changes shape when
narrowed — `dst_transfer` indexes the untouched input layout and needs no
change) exists only on the branch that could not run before.


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

(Was not supported on `RivTreeCluster` at the time this entry was written --
see "Added -- `set_output_reach` on `RivTreeCluster`" above, later in the v1
line, for how the transfer-source problem this note describes was resolved.)

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
