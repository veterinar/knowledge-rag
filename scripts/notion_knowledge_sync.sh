#!/bin/bash
# Ежечасная синхронизация Notion → корпус → поколение → сервер.
# Контракт: docs/criteria-notion-hourly-sync.md (AC-S1…S10). Токен читается
# только в подпроцессе экспорта из owner-only файла (0600) и никуда не пишется.
# VKS_EXPORTER/VKS_BRIDGE — строки команд: word-splitting намеренный.
set -euo pipefail
umask 077
VKS_SECRET_ENV="${VKS_SECRET_ENV:-$HOME/.config/vetclub-secrets/notion.env}"
VKS_EXPORTER="${VKS_EXPORTER:-$HOME/vetpilot/.venv/bin/python $HOME/vetpilot/scripts/notion_export.py}"
VKS_ROOT="${VKS_ROOT:-$HOME/.local/share/vetclub-knowledge-rag/sync}"
VKS_STAGING_DOCS="${VKS_STAGING_DOCS:-$HOME/.local/share/vetclub-knowledge-rag/staging-reseal-20260828-pages/docs}"
VKS_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VKS_PYTHON="${VKS_PYTHON:-/usr/bin/python3}"  # мост — только stdlib
VKS_BRIDGE="${VKS_BRIDGE:-$VKS_PYTHON $VKS_SCRIPT_DIR/build_notion_corpus.py}"
VKS_GEN_CLI="${VKS_GEN_CLI:-$HOME/.local/share/vetclub-knowledge-rag-runtimes/4.9.1-7d47b03/venv/bin/knowledge-rag-generation}"
KNOWLEDGE_RAG_DIR="${KNOWLEDGE_RAG_DIR:-$HOME/.local/share/vetclub-knowledge-rag/staging-reseal-20260828-pages-7d47b03/config}"
export KNOWLEDGE_RAG_DIR
VKS_LAUNCHD_LABEL="${VKS_LAUNCHD_LABEL:-com.vetclub.knowledge-rag}"
VKS_MIN_FREE_GIB="${VKS_MIN_FREE_GIB:-20}"
VKS_FREE_GIB_OVERRIDE="${VKS_FREE_GIB_OVERRIDE:-}"
VKS_VERIFY_TIMEOUT="${VKS_VERIFY_TIMEOUT:-60}"
# Просроченный лок: 21600 с = 10× максимальная измеренная сборка (830 с), округлено до часов.
VKS_LOCK_STALE_S="${VKS_LOCK_STALE_S:-21600}"
VKS_REPO_SNAPSHOT="${VKS_REPO_SNAPSHOT:-$HOME/vetpilot/knowledge/notion}"
SNAP="$VKS_ROOT/snapshot"; SNAP_NEW="$VKS_ROOT/snapshot.new"
mkdir -p "$VKS_ROOT" "$VKS_STAGING_DOCS"
LOG="$VKS_ROOT/sync.log"; RECEIPTS="$VKS_ROOT/receipts.jsonl"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(now)" "$*" >>"$LOG"; }
receipt() { printf '%s\n' "$1" >>"$RECEIPTS"; printf '%s\n' "$1"; }
server_pid() { launchctl print "gui/$(id -u)/$VKS_LAUNCHD_LABEL" 2>/dev/null | awk '/^[[:space:]]*pid = /{print $3}' || true; }
STEP="init"
# shellcheck disable=SC2154  # code присваивается в trap
trap 'code=$?; [ "$code" -ne 0 ] && receipt "{\"error\":\"unexpected\",\"step\":\"$STEP\",\"code\":$code}"' ERR

# 1. lock: mkdir-лок с pid; просроченный (pid мёртв и возраст ≥ VKS_LOCK_STALE_S) снимается
STEP="lock"
if ! mkdir "$VKS_ROOT/lock" 2>/dev/null; then
  STALE_PID=$(cat "$VKS_ROOT/lock/pid" 2>/dev/null || true)
  LOCK_AGE=$(( $(date +%s) - $(stat -f %m "$VKS_ROOT/lock") ))
  if [ -n "$STALE_PID" ] && ! kill -0 "$STALE_PID" 2>/dev/null && [ "$LOCK_AGE" -ge "$VKS_LOCK_STALE_S" ]; then
    rm -rf "$VKS_ROOT/lock"; log "lock: stale removed pid=$STALE_PID age=${LOCK_AGE}s"; mkdir "$VKS_ROOT/lock"
  else
    log "skip: locked pid=${STALE_PID:-<нет>} age=${LOCK_AGE}s"; receipt '{"skip":"locked"}'; exit 0
  fi
fi
trap 'rm -rf "$VKS_ROOT/lock"' EXIT
echo $$ > "$VKS_ROOT/lock/pid"; log "lock: acquired"

# секрет: owner-only 0600; отказ до любого экспорта
STEP="secret"
if [ ! -f "$VKS_SECRET_ENV" ] || [ "$(stat -f %Lp "$VKS_SECRET_ENV" 2>/dev/null || echo 000)" != "600" ]; then
  log "secret: отсутствует или права != 600: $VKS_SECRET_ENV"; receipt '{"error":"secret_env"}'; exit 1
fi

# 2. export (токен живёт только внутри подпроцесса)
STEP="export"
T0=$(date +%s); rm -rf "$SNAP_NEW"; mkdir -p "$SNAP_NEW"
if ! (set +u; set -a; . "$VKS_SECRET_ENV"; set +a; exec $VKS_EXPORTER --out "$SNAP_NEW") >>"$LOG" 2>&1; then
  log "export: ошибка"; receipt '{"error":"export"}'; exit 1
fi
T_EXPORT=$(date +%s); log "export: ok"

# 3. normalize + digest: без строк времени/провенанса; относительные имена
#    файлов в дайджесте — переименование видно
STEP="digest"
snapshot_digest() {
  { find "$1" -maxdepth 1 -type f -name '*.json' | LC_ALL=C sort
    find "$1/pages" -type f -name '*.md' 2>/dev/null | LC_ALL=C sort; } \
  | while IFS= read -r f; do
      printf '%s\n' "${f#"$1"/}"
      grep -vE '^[[:space:]]*("exported_at"|"source_commit"|exported_at:|source_commit:)' "$f" || true
    done | shasum -a 256 | awk '{print $1}'
}
DIG_NEW=$(snapshot_digest "$SNAP_NEW")
if [ -d "$SNAP" ]; then DIG_OLD=$(snapshot_digest "$SNAP"); else DIG_OLD=""; fi
log "digest: new=$DIG_NEW old=${DIG_OLD:-<нет>}"
# AC-S8: сравнение с репозиторным снимком (только чтение, в репозиторий не пишем)
if [ -d "$VKS_REPO_SNAPSHOT" ]; then
  if [ "$(snapshot_digest "$VKS_REPO_SNAPSHOT")" = "$DIG_NEW" ]; then REPO_SNAP="same"; else REPO_SNAP="behind"; fi
else REPO_SNAP="absent"; fi

# 4. diff
STEP="diff"
if [ -n "$DIG_OLD" ] && [ "$DIG_OLD" = "$DIG_NEW" ]; then
  receipt "{\"unchanged\":true,\"digest\":\"$DIG_NEW\",\"repo_snapshot\":\"$REPO_SNAP\"}"
  rm -rf "$SNAP_NEW"; log "diff: без изменений (repo_snapshot=$REPO_SNAP)"; exit 0
fi

# 5. gate: свободное место на / (ГиБ)
STEP="gate"
if [ -n "$VKS_FREE_GIB_OVERRIDE" ]; then FREE_GIB="$VKS_FREE_GIB_OVERRIDE"
else FREE_GIB=$(df -g / | awk 'NR==2 {print $4}'); fi
if [ "${FREE_GIB%.*}" -lt "$VKS_MIN_FREE_GIB" ]; then
  receipt "{\"skip\":\"disk_low\",\"free_gib\":$FREE_GIB}"; log "gate: disk_low free=${FREE_GIB}GiB, снимок не принят"; exit 0
fi
log "gate: free=${FREE_GIB}GiB >= $VKS_MIN_FREE_GIB"

# 5b. корпус build читает тот же каталог: documents_dir из config.yaml
STEP="staging_check"
if ! grep -Eq "^[[:space:]]*documents_dir:[[:space:]]*[\"']?$VKS_STAGING_DOCS[\"']?[[:space:]]*$" "$KNOWLEDGE_RAG_DIR/config.yaml"; then
  log "staging: documents_dir в config.yaml != '$VKS_STAGING_DOCS'"; receipt '{"error":"staging_mismatch"}'; exit 1
fi

# 6. bridge → временный корпус → замена notion-vet + staged-manifest (-print0/-0: пробелы в именах)
STEP="bridge"
DIG12=${DIG_NEW:0:12}; TMP_CORPUS="$VKS_ROOT/notion-vet.tmp"
rm -rf "$TMP_CORPUS"
if ! $VKS_BRIDGE --snapshot-dir "$SNAP_NEW" --out "$TMP_CORPUS" --source-commit "sync-$DIG12" >>"$LOG" 2>&1; then
  log "bridge: ошибка"; receipt '{"error":"bridge"}'; exit 1
fi
if [ -d "$VKS_STAGING_DOCS/notion-vet" ]; then
  rm -rf "$VKS_ROOT/notion-vet.prev"; mv "$VKS_STAGING_DOCS/notion-vet" "$VKS_ROOT/notion-vet.prev"
fi
mv "$TMP_CORPUS" "$VKS_STAGING_DOCS/notion-vet"
STEP="manifest"
if ! ( cd "$VKS_STAGING_DOCS" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 ) \
  > "$VKS_STAGING_DOCS/../staged-manifest.sha256" || [ ! -s "$VKS_STAGING_DOCS/../staged-manifest.sha256" ]; then
  log "manifest: пустой список или ошибка"; receipt '{"error":"manifest_empty"}'; exit 1
fi
T_BRIDGE=$(date +%s); log "bridge: ok sync-$DIG12, manifest пересобран"

# 7. build
STEP="build"
GEN_ID="gen-sync-$(date -u +%Y%m%dT%H%MZ)-$DIG12"
if ! KNOWLEDGE_RAG_DIR="$KNOWLEDGE_RAG_DIR" "$VKS_GEN_CLI" build --generation-id "$GEN_ID" >>"$LOG" 2>&1; then
  log "build: ошибка $GEN_ID"; receipt '{"error":"build"}'; exit 1
fi
T_BUILD=$(date +%s); log "build: ok $GEN_ID"

# 8. restart + verify: servable=true, указатель == GEN_ID, смена pid сервера
#    и живой процесс через 5 с после появления нового pid
STEP="verify"
PID_BEFORE=$(server_pid)
if ! launchctl kickstart -k "gui/$(id -u)/$VKS_LAUNCHD_LABEL" >>"$LOG" 2>&1; then
  log "verify: kickstart ошибка"; receipt '{"error":"kickstart"}'; exit 1
fi
WAITED=0; SERVED=""
while [ "$WAITED" -lt "$VKS_VERIFY_TIMEOUT" ]; do
  ST=$(KNOWLEDGE_RAG_DIR="$KNOWLEDGE_RAG_DIR" "$VKS_GEN_CLI" status 2>>"$LOG" || true)
  PID_AFTER=$(server_pid)
  if printf '%s' "$ST" | grep -Eq '"servable": *true' && printf '%s' "$ST" | grep -q "$GEN_ID" \
     && [ -n "$PID_AFTER" ] && [ "$PID_AFTER" != "$PID_BEFORE" ]; then
    sleep 5
    if kill -0 "$PID_AFTER" 2>/dev/null; then SERVED=1; break; fi
  fi
  sleep 2; WAITED=$((WAITED + 2))
done
if [ -z "$SERVED" ]; then
  log "verify: сервер не на $GEN_ID (pid ${PID_BEFORE:-?}->${PID_AFTER:-?})"; receipt '{"error":"server_pointer"}'; exit 1
fi
T_VERIFY=$(date +%s); log "verify: servable, generation $GEN_ID, pid ${PID_BEFORE:-?}->${PID_AFTER:-?}"

# 9. accept: снимок становится принятым, итоговая квитанция
STEP="accept"
rm -rf "$SNAP"; mv "$SNAP_NEW" "$SNAP"
receipt "{\"ts\":\"$(now)\",\"digest_before\":\"$DIG_OLD\",\"digest_after\":\"$DIG_NEW\",\"generation_id\":\"$GEN_ID\",\"repo_snapshot\":\"$REPO_SNAP\",\"seconds\":{\"export\":$((T_EXPORT - T0)),\"bridge\":$((T_BRIDGE - T_EXPORT)),\"build\":$((T_BUILD - T_BRIDGE)),\"verify\":$((T_VERIFY - T_BUILD))}}"
log "accept: $GEN_ID (repo_snapshot=$REPO_SNAP)"
