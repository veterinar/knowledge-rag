# Acceptance: content-bound freshness fast path

## Problem

Versioned serving currently recomputes the complete content-bound freshness
proof for every read and again after materialization.  That proof re-hashes the
live corpus, source/runtime evidence and sealed generation artifacts and also
recomputes the Chroma/FTS row universes.  The Vault Tool Plane canary therefore
took 282 seconds even though the generation and corpus did not change.

## Required behavior

The first versioned read in a process MUST execute the existing complete
freshness proof.  A later read MAY reuse that successful proof only when a
fresh, deterministic invalidation fingerprint is byte-for-byte identical to
the fingerprint captured around the complete proof.

The fingerprint MUST cover every mutable filesystem input whose contents are
read by the complete proof, including:

- the current pointer and pinned generation receipt;
- the sealed generation artifacts and backend evidence;
- every admitted live-corpus entry and the admitted path set;
- installed package/runtime evidence, dependency lock and package source;
- retrieval configuration and configured model/reranker artifacts.

For every covered entry the fingerprint MUST bind its relative path, entry
type, device, inode, mode, size, `mtime_ns` and `ctime_ns`.  Symlinks, special
files, missing roots and unreadable evidence fail closed.  The fingerprint
MUST NOT contain absolute paths, file contents or secrets.

This is an invalidation-bound cache, not a TTL cache.  Elapsed time alone MUST
never authorize reuse.

## Ordering and concurrency

1. Capture fingerprint A.
2. If A equals the last accepted fingerprint, the warm gate succeeds without
   running the complete proof.
3. Otherwise run the existing complete proof unchanged.
4. Capture fingerprint B after the complete proof.
5. Accept/cache only when the complete proof is GREEN and A equals B.
6. If A differs from B, fail closed with a sanitized environment-changed
   reason and do not cache.

Only one thread may populate the accepted fingerprint at a time.  Concurrent
waiters must re-evaluate the fingerprint after acquiring the lock.  A stale or
unreadable observation clears the accepted fingerprint.

The post-materialization gate MUST take a fresh fingerprint.  A mutation that
lands during retrieval therefore invalidates the warm proof, triggers the full
proof and discards the result on drift.

## Focused verification

The candidate is accepted only when targeted tests prove:

1. the first healthy check invokes the complete proof once and an unchanged
   second check uses the warm fast path;
2. an admitted corpus edit invalidates the cache, including a same-size edit
   after restoring `mtime` (the `ctime` binding must still change);
3. adding or deleting an admitted corpus path invalidates the cache;
4. pointer, receipt or sealed-artifact mutation invalidates the cache;
5. package/config/lock/model evidence mutation invalidates the cache;
6. a fingerprint change during complete verification fails closed and is not
   cached;
7. the existing mid-query drift test still discards the result;
8. legacy mode remains unchanged.

Use only the focused generation-integration tests needed for these contracts;
do not run the full suite during authoring.

## Performance acceptance

After correctness is GREEN, run one bounded local canary without rebuilding an
index or loading a new model:

- one cold `get_index_stats` establishes the full-proof cost;
- five unchanged warm `get_index_stats` calls must each finish in at most 10%
  of the cold call duration and no slower than 2 seconds;
- one fixed FTS read, including its pre/post freshness gates, must finish in
  at most 5 seconds;
- the process remains healthy and the generation/receipt identity is unchanged.

If either absolute threshold is not met, the profile remains explicit opt-in.
No production runtime is changed by this candidate.

## Candidate evidence (2026-08-22)

The focused executable check passed 8/8 cases in 1.51 seconds.  The same warm
reuse test overlaid on the exact unfixed base
`bb88bb7ae7fa3dd7325544c693e60a11a330b661` was RED for the intended reason:
the backend proof ran twice where the candidate runs it once (`2 != 1`).

A bounded read-only source canary used the installed 4.9.1 runtime dependencies,
the current sealed generation
`gen-5c6f941280c249f08f3e5fed0cddc5d7` and receipt
`4e7e2b8f715c97f934c72ba1f14b414ac7cd5edf027b684d0265ae1e8b5b0cda`.
The existing receipt necessarily names the pre-candidate source digest, so the
cold source canary substituted only that receipt's existing `code_sha256`;
all content, corpus, configuration, model, dependency and backend proof paths
otherwise ran unchanged.  This is candidate performance evidence, not a
deployed-runtime receipt.

- startup pin: 8.703 seconds;
- cold complete gate: 5.101 seconds;
- five unchanged warm gates: 0.458, 0.456, 0.456, 0.500 and 0.475 seconds;
- worst warm/cold ratio: 9.80%; every warm gate was below 2 seconds;
- one sealed FTS read with warm pre/post gates: 0.908 seconds, one hit;
- five direct metadata captures produced one identical digest in
  0.260, 0.228, 0.250, 0.227 and 0.227 seconds.

Host observations before/after the bounded canary: no thermal or performance
warning, free memory 59% -> 60%, task-local write growth 0 KiB and more than
65 GiB disk free.  Swap was unavailable to the sandboxed probe and remains
unmeasured.

Default activation still waits for merge, a new immutable runtime whose receipt
binds the candidate source bytes, and the same bounded canary without the
source-identity substitution.

## Out of scope

- no TTL-only trust window;
- no removal or weakening of the complete content/digest/backend proof;
- no watcher, index rebuild, model change or retrieval-scoring change;
- no Tool Plane URL/factory override, mutation tool or central proxy;
- no default Vault-profile activation until this candidate is merged,
  deployed as a new immutable runtime and the bounded canary is GREEN.
