#!/bin/bash
# Стендовые пробы S3, S9-change, S9-lock, S4, S6, S5 для
# scripts/notion_knowledge_sync.sh (AC-S9, docs/criteria-notion-hourly-sync.md).
# Только sh + python3-набор утилит, без сети, без реальных launchctl/моста/сборки.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SYNC="$SCRIPT_DIR/../scripts/notion_knowledge_sync.sh"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/vks-sync-test.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
ROOT="$TMP/root"; DOCS="$TMP/docs"; BIN="$TMP/bin"; EXPORTS="$TMP/exports"
mkdir -p "$ROOT" "$DOCS" "$BIN" "$EXPORTS"

CANARY="TOKEN-CANARY-$RANDOM$RANDOM"
printf 'NOTION_TOKEN=%s\n' "$CANARY" > "$TMP/notion.env"
chmod 600 "$TMP/notion.env"

# --- заглушки --------------------------------------------------------------
# Экспортёр: фиксированный снимок; exported_at/source_commit меняются каждый
# вызов; содержательное поле «Установлено» и тело страницы — из $VKS_TEST_MARKER.
cat > "$BIN/exporter" <<EOF
#!/bin/bash
OUT="\$2"; STAMP="\$(date +%s)\$\$"; MARK="\$VKS_TEST_MARKER"
mkdir -p "\$OUT/pages"
printf '{"databases":{"pravila":{"sha256":"x%s","database_id":"db1"}},"pages":{"p":{"sha256":"y%s","page_id":"pg1","title":"P"}}}' "\$MARK" "\$MARK" > "\$OUT/manifest.json"
printf '[{"id":"r1","Название":"Правило","Установлено":"%s — длинное тело записи, чтобы попасть в корпус как содержательный текст записи pravila"}]' "\$MARK" > "\$OUT/pravila.json"
printf 'exported_at: %s\nsource_commit: %s\n# страница p\nтело страницы %s\n' "\$STAMP" "\$STAMP" "\$MARK" > "\$OUT/pages/p.md"
EOF
# Мост: «строит» корпус, пишет факт вызова.
cat > "$BIN/bridge" <<EOF
#!/bin/bash
# argv: --snapshot-dir D --out O --source-commit C  → O = \$4
mkdir -p "\$4"; echo x > "\$4/f.md"
echo "\$*" >> "$EXPORTS/bridge.calls"
EOF
# Сборка/статус: build пишет факт и generation_id; status отдаёт последний
# построенный (или \$VKS_STATUS_GEN, если задан — проба S5).
cat > "$BIN/genc" <<EOF
#!/bin/bash
case "\$1" in
  build) echo "\$3" >> "$EXPORTS/build.calls"; echo "\$3" >> "$EXPORTS/built.gen" ;;
  status) G=\$(tail -n 1 "$EXPORTS/built.gen" 2>/dev/null || true); [ -n "\$VKS_STATUS_GEN" ] && G="\$VKS_STATUS_GEN"; printf '{"servable": true, "generation_id": "%s"}\n' "\$G" ;;
esac
EOF
# launchctl в начале PATH: пишет факт вызова.
cat > "$BIN/launchctl" <<EOF
#!/bin/bash
echo "\$*" >> "$EXPORTS/launchctl.calls"
EOF
chmod +x "$BIN/exporter" "$BIN/bridge" "$BIN/genc" "$BIN/launchctl"

n_calls() { if [ -f "$EXPORTS/$1.calls" ]; then wc -l < "$EXPORTS/$1.calls" | tr -d ' '; else echo 0; fi; }
last_receipt() { tail -n 1 "$ROOT/receipts.jsonl"; }
reset_calls() { rm -f "$EXPORTS/bridge.calls" "$EXPORTS/build.calls" "$EXPORTS/launchctl.calls"; }

run_sync() {
  VKS_SECRET_ENV="$TMP/notion.env" \
  VKS_EXPORTER="$BIN/exporter" \
  VKS_BRIDGE="$BIN/bridge" \
  VKS_GEN_CLI="$BIN/genc" \
  VKS_STAGING_DOCS="$DOCS" \
  VKS_ROOT="$ROOT" \
  VKS_TEST_MARKER="${VKS_TEST_MARKER:-base}" \
  VKS_VERIFY_TIMEOUT=4 \
  PATH="$BIN:$PATH" \
  bash "$SYNC" >/dev/null 2>&1 || true
}

FAILED=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; FAILED=1; fi; }

# --- S9-lock: существующий lock/ → skip:locked, ничего не вызвано -----------
mkdir -p "$ROOT/lock"
reset_calls
run_sync
check "S9-lock receipt" '[ "$(last_receipt)" = "{\"skip\":\"locked\"}" ]'
check "S9-lock no-calls" '[ "$(n_calls bridge)" = 0 ] && [ "$(n_calls build)" = 0 ] && [ "$(n_calls launchctl)" = 0 ]'
rmdir "$ROOT/lock"

# --- прогон 1: принятие базового снимка -------------------------------------
VKS_TEST_MARKER=base run_sync
check "run1 receipt-gen" 'last_receipt | grep -q generation_id'
reset_calls

# --- S3: прогон 2 без содержательных изменений ------------------------------
VKS_TEST_MARKER=base run_sync
check "S3 unchanged" 'last_receipt | grep -q "\"unchanged\":true"'
check "S3 no-calls" '[ "$(n_calls bridge)" = 0 ] && [ "$(n_calls build)" = 0 ] && [ "$(n_calls launchctl)" = 0 ]'

# --- S9-change: прогон 3 с изменённым полем ---------------------------------
reset_calls
VKS_TEST_MARKER=changed run_sync
check "S9-change bridge" '[ "$(n_calls bridge)" -ge 1 ]'
check "S9-change build" '[ "$(n_calls build)" -ge 1 ]'
check "S9-change kickstart" '[ "$(n_calls launchctl)" -ge 1 ]'
check "S9-change receipt" 'last_receipt | grep -q generation_id'
check "S9-change snapshot" '[ -d "$ROOT/snapshot" ] && [ ! -e "$ROOT/snapshot.new" ]'
check "S9-change corpus" '[ -f "$DOCS/notion-vet/f.md" ] && [ -f "$DOCS/../staged-manifest.sha256" ]'

# --- S4: 5 ГиБ свободно при изменении → skip:disk_low, снимок не принят -----
B4=$(n_calls build)
VKS_TEST_MARKER=changed2 VKS_FREE_GIB_OVERRIDE=5 run_sync
check "S4 receipt" 'last_receipt | grep -q "disk_low"'
check "S4 no-build" "[ \"\$(n_calls build)\" = \"$B4\" ]"
check "S4 snapshot-not-accepted" '! grep -q changed2 "$ROOT/snapshot/pravila.json" 2>/dev/null'

# --- S6: значение NOTION_TOKEN не течёт в лог и квитанции -------------------
check "S6 log" '! grep -q "$CANARY" "$ROOT/sync.log"'
check "S6 receipts" '! grep -q "$CANARY" "$ROOT/receipts.jsonl"'

# --- S5: сервер с чужим generation_id → error:server_pointer, код 1 ---------
set +e
VKS_SECRET_ENV="$TMP/notion.env" VKS_EXPORTER="$BIN/exporter" \
VKS_BRIDGE="$BIN/bridge" VKS_GEN_CLI="$BIN/genc" \
VKS_STAGING_DOCS="$DOCS" VKS_ROOT="$ROOT" \
VKS_TEST_MARKER=changed3 VKS_STATUS_GEN="gen-other-9999" \
VKS_VERIFY_TIMEOUT=4 PATH="$BIN:$PATH" \
bash "$SYNC" >/dev/null 2>&1
RC=$?
set -e
check "S5 receipt" 'last_receipt | grep -q server_pointer'
check "S5 exit1" "[ \"$RC\" = 1 ]"
check "S5 lock-released" '[ ! -e "$ROOT/lock" ]'

exit "$FAILED"
