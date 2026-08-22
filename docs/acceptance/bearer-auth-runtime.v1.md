# Bearer authentication for the local knowledge-rag runtime

Status: implementation contract for an isolated candidate. Production activation is a separate runtime action.

## Outcome

The existing streamable-HTTP service keeps its fixed loopback listener and requires a static bearer credential on every MCP request. `/health` remains credential-free for the LaunchAgent probe. The credential is held in one owner-only regular file outside Git and is never copied into YAML, process arguments, logs, errors, receipts, or result payloads.

The server and the bundled `vault-rag` client read the same credential file. A missing, empty, malformed, symlinked, non-owner, or group/world-accessible file fails closed before an MCP request is served or sent.

## Configuration contract

`server.auth` accepts exactly one of:

```yaml
auth:
  bearer_token: "legacy-inline-token"
```

or:

```yaml
auth:
  bearer_token_file: "/absolute/owner-only/path"
```

The two fields are mutually exclusive. `bearer_token_file` must be an absolute path to a regular non-symlink file owned by the current user, with no group/world permission bits. Its content is one 32–512 byte ASCII RFC 6750 `token68` value after one optional terminal newline: `[A-Za-z0-9-._~+/]+` followed only by optional trailing `=` padding. Non-ASCII input, embedded whitespace, NUL, embedded padding, malformed data, or additional lines are rejected. The resolved token is stored only in the runtime-only `auth_bearer_token` field whose representation and comparisons omit it.

The legacy inline field remains compatible for existing deployments, but the VetClub runtime uses only `bearer_token_file`.

## Client contract

The bundled client reads `KNOWLEDGE_RAG_BEARER_TOKEN_FILE` through the same validator. When present it constructs the pinned streamable-HTTP transport with an `Authorization: Bearer …` default header. It never appends the token to the URL and never prints the token when connection or authentication fails. When the environment variable is absent, current unauthenticated local-client behaviour remains available for legacy runtimes.

## Runtime rollout

1. Create one random token file outside repositories and Vault, directory mode `0700`, file mode `0400` or `0600`.
2. Put only the token-file path in the immutable runtime config and in the client LaunchAgent environment.
3. Restart once, retaining the previous runtime and config as rollback.
4. Require: `/health` returns healthy without credentials; `/mcp` rejects missing and wrong tokens; an authenticated MCP initialize/list call succeeds; listener remains exactly `127.0.0.1`.
5. On any failure restore the previous plist/runtime pointer once and stop. Do not expose a non-loopback listener in this slice.

## Focused acceptance

Two focused modules cover the following eight behaviours. Parameterised invalid-file cases count as separate test outcomes but do not expand the contract:

1. valid owner-only token file resolves and the dataclass representation omits it;
2. inline token remains supported;
3. inline plus file is rejected;
4. relative, missing, symlinked, non-regular, empty, multiline, short, or overlong file is rejected;
5. group/world permissions are rejected;
6. client transport receives the bearer header without putting it in the URL;
7. token-file and arbitrary connection errors use fixed messages and contain no credential, URL, path, exception text, or traceback;
8. no-token legacy client path remains unchanged.

One black-box config case is observed RED on the exact base and GREEN on the candidate. Locally run only the two affected test files, the existing package-data identity check, and changed-file lint; no full suite.

## Out of scope

- changing `127.0.0.1`, firewall, router, DNS, TLS, reverse proxy, or public exposure;
- OAuth, multi-user identities, token rotation service, or a central credential vault;
- placing credential bytes in Git, Vault, YAML, plist, command arguments, receipts, or logs;
- commit, push, PR, merge, immutable-runtime build, restart, or activation without their separate authority.
