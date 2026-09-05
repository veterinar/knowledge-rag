#!/bin/bash
# Стендовые пробы S3, S9-change, S9-lock, S4, S6(+perm), S5(+kickstart-fail),
# S3-export-fail, S8-repo для scripts/notion_knowledge_sync.sh
# (AC-S9, docs/criteria-notion-hourly-sync.md).
# Только sh + python3-набор утилит, без сети, без реальных launchctl/моста/сборки.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SYNC="$SCRIPT_DIR/../scripts/notion_knowledge_sync.sh"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/vks-sync-test.XXXXXX")
LIVE_PID=""
cleanup() { [ -n "$LIVE_PID" ] && kill "$LIVE_PID" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT
ROOT="$TMP/root"; DOCS="$TMP/docs"; BIN="$TMP/bin"; EXPORTS="$TMP/exports"
KRD="$TMP/krd"
mkdir -p "$ROOT" "$DOCS" "$BIN" "$EXPORTS" "$KRD"

# config сборки: documents_dir указывает на стендовый staging
printf 'model: bge-small\nchunking:\n  size: 512\ndocuments_dir: "%s"\n' "$DOCS" > "$KRD/config.yaml"

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
# Экспортёр-отказ: проба S3-export-fail.
cat > "$BIN/exporter-fail" <<'EOF'
#!/bin/bash
exit 1
EOF
# Мост: «строит» корпус, пишет факт вызова.
cat > "$BIN/bridge" <<EOF
#!/bin/bash
# argv: --snapshot-dir D --out O --source-commit C  → O = \$4
mkdir -p "\$4"; echo x > "\$4/f.md"
echo "\$*" >> "$EXPORTS/bridge.calls"
EOF
# Сборка/статус: build пишет факт и generation_id; status требует
# KNOWLEDGE_RAG_DIR в окружении (как реальный CLI) и отдаёт последний
# построенный (или \$VKS_STATUS_GEN, если задан — проба S5).
cat > "$BIN/genc" <<EOF
#!/bin/bash
case "\$1" in
  build) echo "\$3" >> "$EXPORTS/build.calls"; echo "\$3" >> "$EXPORTS/built.gen" ;;
  status) [ -z "\$KNOWLEDGE_RAG_DIR" ] && { echo legacy >&2; exit 2; }
    G=\$(tail -n 1 "$EXPORTS/built.gen" 2>/dev/null || true); [ -n "\${VKS_STATUS_GEN:-}" ] && G="\${VKS_STATUS_GEN:-}"; printf '{"servable": true, "generation_id": "%s", "receipt_sha256": "stub-receipt-sha"}\n' "\$G" ;;
esac
EOF
# launchctl: kickstart печатает факт; print отдаёт pid живого процесса.
# Счётчик в файле: каждый kickstart выдаёт новый pid. До первого kickstart —
# pid самого теста (\$\$), после — pid фонового sleep-процесса стенда.
sleep 300 &
LIVE_PID=$!
cat > "$BIN/launchctl" <<EOF
#!/bin/bash
case "\$1" in
  kickstart) echo "\$*" >> "$EXPORTS/launchctl.calls"
    [ -f "$EXPORTS/kick.count" ] || echo 0 > "$EXPORTS/kick.count"
    N=\$(( \$(cat "$EXPORTS/kick.count") + 1 )); echo "\$N" > "$EXPORTS/kick.count"
    [ "\${VKS_LAUNCHCTL_FAIL:-0}" = 1 ] && exit 1
    exit 0 ;;
  print) if [ "\${VKS_LAUNCHCTL_STUCK:-0}" = 1 ]; then echo "  pid = $LIVE_PID"; exit 0; fi
    N=0; [ -f "$EXPORTS/kick.count" ] && N=\$(cat "$EXPORTS/kick.count") || N=0
    if [ \$(( N % 2 )) -eq 1 ]; then echo "  pid = $LIVE_PID"; else echo "  pid = $$"; fi
    exit 0 ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$BIN/exporter" "$BIN/exporter-fail" "$BIN/bridge" "$BIN/genc" "$BIN/launchctl"

n_calls() { if [ -f "$EXPORTS/$1.calls" ]; then wc -l < "$EXPORTS/$1.calls" | tr -d ' '; else echo 0; fi; }
last_receipt() { tail -n 1 "$ROOT/receipts.jsonl"; }
reset_calls() { rm -f "$EXPORTS/bridge.calls" "$EXPORTS/build.calls" "$EXPORTS/launchctl.calls"; }

run_sync() {
  VKS_SECRET_ENV="${VKS_SECRET_ENV:-$TMP/notion.env}" \
  VKS_REPO_SNAPSHOT="${VKS_REPO_SNAPSHOT:-$TMP/repo-snap-nonexistent}" \
  VKS_EXPORTER="$BIN/exporter" \
  VKS_BRIDGE="$BIN/bridge" \
  VKS_GEN_CLI="$BIN/genc" \
  VKS_STAGING_DOCS="$DOCS" \
  VKS_ROOT="$ROOT" \
  KNOWLEDGE_RAG_DIR="$KRD" \
  VKS_TEST_MARKER="${VKS_TEST_MARKER:-base}" \
  VKS_VERIFY_TIMEOUT=20 \
  PATH="$BIN:$PATH" \
  bash "$SYNC" >/dev/null 2>&1 || true
}

FAILED=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; FAILED=1; fi; }

# --- S9-lock: существующий lock/ → skip:locked, ничего не вызвано -----------
mkdir -p "$ROOT/lock"
reset_calls
run_sync
check "S9-lock receipt" 'last_receipt | grep -Eq "^\{\"ts\":\"[^\"]+\",\"skip\":\"locked\"\}$"'
check "S9-lock no-calls" '[ "$(n_calls bridge)" = 0 ] && [ "$(n_calls build)" = 0 ] && [ "$(n_calls launchctl)" = 0 ]'
# детектор регрессии r3: при чужом локе сценарий не должен снимать каталог lock
check "S9-lock-kept" '[ -d "$ROOT/lock" ]'
rm -rf "$ROOT/lock"

# --- S9-lock-nopid: просроченный лок без файла pid снимается -----------------
mkdir -p "$ROOT/lock"
reset_calls
VKS_TEST_MARKER=nopid VKS_LOCK_STALE_S=0 run_sync
check "S9-lock-nopid not-skipped" '! (last_receipt | grep -q "\"skip\":\"locked\"")'
check "S9-lock-nopid lock-removed" '[ ! -d "$ROOT/lock" ]'
rm -rf "$ROOT/lock" 2>/dev/null || true

# --- прогон 1: принятие базового снимка -------------------------------------
VKS_TEST_MARKER=base run_sync
check "run1 receipt-gen" 'last_receipt | grep -q generation_id'
check "run1 receipt-sha" 'last_receipt | grep -q "\"receipt_sha256\":\"stub-receipt-sha\""'
check "run1 receipt-ts" 'last_receipt | grep -Eq "^\{\"ts\":"'
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
check "S4 receipt" 'last_receipt | grep -q disk_low'
check "S4 no-build" "[ \"\$(n_calls build)\" = \"$B4\" ]"
check "S4 snapshot-not-accepted" '! grep -q changed2 "$ROOT/snapshot/pravila.json" 2>/dev/null'

# --- S6: значение NOTION_TOKEN не течёт в лог и квитанции -------------------
check "S6 log" '! grep -q "$CANARY" "$ROOT/sync.log"'
check "S6 receipts" '! grep -q "$CANARY" "$ROOT/receipts.jsonl"'

# --- S6-perm: секрет 644 → error:secret_env, экспорт не вызван --------------
B6=$(n_calls build)
cp "$TMP/notion.env" "$TMP/notion-644.env"; chmod 644 "$TMP/notion-644.env"
rm -rf "$ROOT/snapshot.new"
VKS_TEST_MARKER=changed3 VKS_SECRET_ENV="$TMP/notion-644.env" run_sync
check "S6-perm receipt" 'last_receipt | grep -q secret_env'
check "S6-perm no-export" '[ ! -e "$ROOT/snapshot.new/pravila.json" ]'
check "S6-perm no-build" "[ \"\$(n_calls build)\" = \"$B6\" ]"

# --- S3-export-fail: экспортёр отказал → error:export, лок снят -------------
rm -rf "$ROOT/snapshot.new"; rmdir "$ROOT/lock" 2>/dev/null || true
set +e
VKS_SECRET_ENV="$TMP/notion.env" VKS_EXPORTER="$BIN/exporter-fail" \
VKS_BRIDGE="$BIN/bridge" VKS_GEN_CLI="$BIN/genc" \
VKS_STAGING_DOCS="$DOCS" VKS_ROOT="$ROOT" KNOWLEDGE_RAG_DIR="$KRD" \
VKS_TEST_MARKER=changed4 VKS_VERIFY_TIMEOUT=20 PATH="$BIN:$PATH" \
bash "$SYNC" >/dev/null 2>&1
RC3=$?
check "S3-export-fail receipt" 'last_receipt | grep -q "\"error\":\"export\""'
check "S3-export-fail exit1" "[ \"$RC3\" = 1 ]"
check "S3-export-fail lock-released" '[ ! -e "$ROOT/lock" ]'

# --- S5-kickstart-fail: launchctl kickstart отказал → error:kickstart --------
rm -f "$EXPORTS/kick.count"; reset_calls
set +e
VKS_SECRET_ENV="$TMP/notion.env" VKS_EXPORTER="$BIN/exporter" \
VKS_BRIDGE="$BIN/bridge" VKS_GEN_CLI="$BIN/genc" \
VKS_STAGING_DOCS="$DOCS" VKS_ROOT="$ROOT" KNOWLEDGE_RAG_DIR="$KRD" \
VKS_TEST_MARKER=changed5 VKS_VERIFY_TIMEOUT=20 VKS_LAUNCHCTL_FAIL=1 \
PATH="$BIN:$PATH" \
bash "$SYNC" >/dev/null 2>&1
RC5K=$?
set -e
check "S5-kickstart-fail receipt" 'last_receipt | grep -q "\"error\":\"kickstart\""'
check "S5-kickstart-fail exit1" "[ \"$RC5K\" = 1 ]"
check "S5-kickstart-fail lock-released" '[ ! -e "$ROOT/lock" ]'

# --- S5: сервер с чужим generation_id → error:server_pointer, код 1 ---------
rm -f "$EXPORTS/kick.count"
set +e
VKS_SECRET_ENV="$TMP/notion.env" VKS_EXPORTER="$BIN/exporter" \
VKS_BRIDGE="$BIN/bridge" VKS_GEN_CLI="$BIN/genc" \
VKS_STAGING_DOCS="$DOCS" VKS_ROOT="$ROOT" KNOWLEDGE_RAG_DIR="$KRD" \
VKS_TEST_MARKER=changed6 VKS_STATUS_GEN="gen-other-9999" \
VKS_VERIFY_TIMEOUT=4 PATH="$BIN:$PATH" \
bash "$SYNC" >/dev/null 2>&1
RC=$?
set -e
check "S5 receipt" 'last_receipt | grep -q server_pointer'
check "S5 exit1" "[ \"$RC\" = 1 ]"
check "S5 lock-released" '[ ! -e "$ROOT/lock" ]'

# --- S5-pid-stuck: pid сервера не сменился после kickstart → error:server_pointer
rm -f "$EXPORTS/kick.count"; reset_calls
set +e
VKS_SECRET_ENV="$TMP/notion.env" VKS_EXPORTER="$BIN/exporter" \
  VKS_BRIDGE="$BIN/bridge" VKS_GEN_CLI="$BIN/genc" \
  VKS_STAGING_DOCS="$DOCS" VKS_ROOT="$ROOT" KNOWLEDGE_RAG_DIR="$KRD" \
  VKS_TEST_MARKER=changed8 VKS_VERIFY_TIMEOUT=4 VKS_LAUNCHCTL_STUCK=1 \
  PATH="$BIN:$PATH" \
  bash "$SYNC" >/dev/null 2>&1
RC5P=$?
set -e
check "S5-pid-stuck receipt" 'last_receipt | grep -Eq "^\{\"ts\":\"[^\"]+\",\"error\":\"server_pointer\"\}$"'
check "S5-pid-stuck exit1" "[ \"$RC5P\" = 1 ]"
check "S5-pid-stuck lock-released" '[ ! -e "$ROOT/lock" ]'

# --- S8-repo: поле repo_snapshot same/behind/absent --------------------------
# (принятый снимок — с маркером changed; сначала принимаем его прогоном,
# затем экспортируем тот же маркер — прогон даёт unchanged)
rm -f "$EXPORTS/kick.count"; reset_calls
VKS_TEST_MARKER=changed7 VKS_REPO_SNAPSHOT="$TMP/repo-snap-nonexistent" run_sync
VKS_TEST_MARKER=changed7 VKS_REPO_SNAPSHOT="$TMP/repo-snap-nonexistent" run_sync
check "S8-repo absent" 'last_receipt | grep -q "\"repo_snapshot\":\"absent\""'
# same: снимок совпадает с принятым (копия $ROOT/snapshot)
cp -R "$ROOT/snapshot" "$TMP/repo-snap"
VKS_TEST_MARKER=changed7 VKS_REPO_SNAPSHOT="$TMP/repo-snap" run_sync
check "S8-repo same" 'last_receipt | grep -q "\"repo_snapshot\":\"same\""'
# behind: в репозиторном снимке иное содержимое
printf 'x' >> "$TMP/repo-snap/pravila.json"
VKS_TEST_MARKER=changed7 VKS_REPO_SNAPSHOT="$TMP/repo-snap" run_sync
check "S8-repo behind" 'last_receipt | grep -q "\"repo_snapshot\":\"behind\""'

exit "$FAILED"
