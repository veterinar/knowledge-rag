# Criteria: source policy in mixed generation receipts

Status: criteria-first, implementation pending.

Classification: HIGH / integration. This changes the immutable generation
receipt and the producer admission boundary used by a mixed Vault + Notion
corpus. The change is limited to this repository; it does not read Notion or
Vault data, run a generation build, restart a server, or change MCP tool
signatures.

Bound base: `master` at
`9cb947e665f34531a62e44b6bbb1ace0551d02df` (tree
`b1c56734c5c042ce99cf8f6a3a5b799bcf97659c`, remote
`https://github.com/veterinar/knowledge-rag.git`).

## Goal

Every newly published mixed generation carries an immutable, content-bound
source policy that lets a consumer distinguish the two admitted namespaces:

- `vault-vet`: `project_identity: null`, reserved category `vault-vet`;
- `notion-vet`: `project_identity: vetpilot`, reserved category `notion-vet`.

The producer must refuse publication when namespace coverage, prefix
boundaries, per-namespace digest/counts, or reserved category mappings do not
agree with the sealed corpus. Existing CAS, receipt-SHA, and pre-publication
abort semantics remain the rollback boundary.

## Contract

The receipt adds one top-level `source_policy` object. Its shape is:

```json
{
  "schema_version": 1,
  "policy_sha256": "<64 lowercase hex>",
  "count": "<runtime nonnegative integer>",
  "manifest": {
    "vault-vet": {
      "path_prefix": "vault-vet/",
      "project_identity": null,
      "category": "vault-vet",
      "count": "<runtime nonnegative integer>",
      "manifest_sha256": "<64 lowercase hex>"
    },
    "notion-vet": {
      "path_prefix": "notion-vet/",
      "project_identity": "vetpilot",
      "category": "notion-vet",
      "count": "<runtime nonnegative integer>",
      "manifest_sha256": "<64 lowercase hex>"
    }
  }
}
```

`manifest` is keyed by exactly those two namespace names. `count` equals the
sum of namespace counts. Each `manifest_sha256` is the existing canonical
corpus-manifest digest over sorted namespace-relative
`(relative POSIX path without the namespace prefix, file SHA-256)` entries.
For example, `vault-vet/x.md` contributes `x.md` to the `vault-vet` digest;
the identity's `path_prefix` separately binds that boundary. `policy_sha256`
is the stable canonical SHA-256 of exactly the sorted identity records
`{namespace, path_prefix, category, project_identity}`; it excludes all
counts and per-namespace manifest digests. The two identity records are the
closed producer preset above, with the literal owner-bound project identity
`vetpilot`. The implementation must use one shared canonical helper rather
than a second ad-hoc digest algorithm.

The receipt remains the only activation authority: `source_policy` is covered
by the assembled receipt bytes, the pointer's `receipt_sha256`, and the
existing CAS swap. No caller-supplied `project_id` or `vault_head` is added.

Compatibility choice: keep `schema_version: 3` as an additive, strictly
validated extension. A legacy v3 receipt that has no reserved namespace path
remains valid with its historical exact keys and is not reinterpreted. A new
receipt may carry `source_policy` only when the sealed admitted corpus proves
the reserved namespace prefixes; a mixed corpus containing both namespaces
must carry the complete policy. Validation of the optional field is strict,
while a new Tool Plane consumer must fail closed on any receipt missing the
policy. No schema-version-3 legacy bytes are rewritten.

Legacy receipts may remain inspectable and usable by legacy unscoped flows,
but a mixed-generation producer cannot publish without `source_policy`.
The producer uses this closed preset with the existing category-mapping
pipeline; it adds no caller-selected project field, auth service, or second
runtime. Generic search remains generic; any hard consumer scope is enforced
by the consumer using the receipt policy.

The existing safe generation/status seam (`inspect_current_receipt` consumed
by `get_index_stats`) may expose only the validated source-policy identities,
dynamic counts, and namespace manifest digests. It must not expose raw paths,
provenance, or caller-selected scope. A malformed or missing policy must not
be marked servable by the new Tool Plane status path; MCP tool signatures stay
unchanged.

## Acceptance criteria

S1. The generation producer derives namespace entries from the same admitted
corpus file set used for the receipt, not from directory counts or an external
sidecar. A path is admitted exactly once; root files, `vault-vet-evil/`,
`notion-vet-evil/`, missing files, duplicate entries, symlinks and special
files fail closed.

S2. The two namespace records have exactly the required identity and reserved
category values. Before expensive population, the parsed category mapping must
cover both reserved prefixes with those exact categories. After population,
the staged `index_metadata.json` source set must equal the sealed corpus
relative-path set exactly: every admitted path has exactly one metadata record,
with no duplicate, missing, extra, or unknown source. A configured category
mapping that resolves any admitted file under either namespace to a different
category is a hard pre-publication failure. The validation is boundary-aware
and does not rely on substring matching.

S3. Receipt validation rejects missing, malformed, extra, reordered-in-a-way
that changes the canonical identity digest, count-mismatched, or
manifest/policy-digest-mismatched `source_policy`. The current generation
pointer is unchanged on every such failure; only the temporary `.building-*`
tree may be removed.

S4. The existing receipt schema, activation, status, rollback, and receipt-SHA
checks continue to work for the new receipt. Public MCP functions and their
parameter lists remain byte-compatible with `tests/test_backwards_compat.py`.

S5. The producer path has no network, Notion, Vault, runtime, restart, or
secret dependency. It only validates the staged corpus and the parsed
generation configuration before population/publish.

S6. The implementation preserves the existing failure distinctions: CAS
conflict leaves a sealed generation for explicit `activate`; uncertain commit
state requires status reread; all ordinary pre-publication failures clean only
staging and leave the current pointer untouched.

S7. The safe status seam exposes only the validated source policy (identity
records plus dynamic counts/manifests) through `get_index_stats`' existing
generation block. Missing or malformed policy cannot mark a generation
servable, and no raw corpus path, provenance, or new MCP parameter is added.

S8. The generation CLI derives the policy from the sealed staged corpus and
passes it into `GenerationStore.publish`; a reserved-namespace generation
cannot silently publish with the optional argument omitted. The producer and
status seams share the same validators and canonical digest helpers.

Corrective implementation notes for this slice:

- Preserve every mandatory legacy v3 key and accept exactly the legacy key set
  or that set plus `source_policy`; keep validated unscoped legacy receipts
  servable while the new Tool Plane path fails closed without valid policy.
- The real `generation_cli._build_command` path must derive and pass policy,
  validate parsed reserved category mappings before population, and compare
  staged metadata sources to the sealed corpus by exact one-to-one equality.
  Duplicate, missing, extra, unknown, absolute, non-admitted, or wrong-category
  sources fail before publish; the existing status seam exposes path-free
  validated policy without changing MCP parameters.
- Delete the entire added source-policy test block (including its module-level
  corpus/helper symbols) and leave exactly one function named
  `test_mixed_generation_source_policy_contract`, at most 100 lines including
  one local helper. Use a local nested-path corpus that creates each parent
  directory, invoke the real `_build_command` with the real `GenerationStore`
  seam (patch only population/evidence/config identity as needed), and assert
  the captured policy. No module-level fixture matrix, parametrization,
  duplicate suite, dead/vacuous assertion, or invalid `dict` construction.
  Do not change public MCP parameter signatures.

## Verification

Exactly one focused producer check is required after implementation: the
repository-native node
`tests/test_generations.py::test_mixed_generation_source_policy_contract`.
It must prove a valid two-namespace policy reaches the `_build_command` publish
seam, and that the exact source/category admission failures and legacy v3
compatibility preserve the pointer. Do not run a suite, build, index, or runtime.

## Rollback and handoff

No live activation is part of this slice. Code rollback is the exact candidate
diff reversal. A published generation rollback remains the existing
`knowledge-rag-generation rollback <generation_id>` CAS operation; a receipt
without a valid source policy must never become the new current pointer.

After the focused check, inspect the exact diff and report base/head/tree,
changed files, test output, receipt/source-policy digest semantics, and any
consumer-facing compatibility note. Publication is separate and may proceed
only if the canonical GitHub gates bind this exact head.
