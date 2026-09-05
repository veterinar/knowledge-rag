#!/bin/bash
# Ежечасная синхронизация Notion → корпус → поколение → сервер.
# Контракт: docs/criteria-notion-hourly-sync.md (AC-S1…S10). Токен читается
# только в подпроцессе экспорта из owner-only файла (0600) и никуда не пишется.
set -euo pipefail

VKS_SECRET_ENV="${VKS_SECRET_ENV:-$HOME/.config/vetclub-secrets/notion.env}"
VKS_EXPORTER="${VKS_EXPORTER:-$HOME/vetpilot/.venv/bin/python $HOME/vetpilot/scripts/notion_export.py}"
VKS_ROOT="${VKS_ROOT:-$HOME/.local/share/vetclub-knowledge-rag/sync}"
VKS_STAGING_DOCS="${VKS_STAGING_DOCS:-$HOME/.local/share/vetclub-knowledge-rag/staging-reseal-20260828-pages/docs}"
VKS_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VKS_BRIDGE="${VKS_BRIDGE:-python3 $VKS_SCRIPT_DIR/build_notion_corpus.py}"
VKS_GEN_CLI="${VKS_GEN_CLI:-$HOME/.local/share/vetclub-knowledge-rag-runtimes/4.9.1-7d47b03/venv/bin/knowledge-rag-generation}"
KNOWLEDGE_RAG_DIR="${KNOWLEDGE_RAG_DIR:-$HOME/.local/share/vetclub-knowledge-rag/staging-reseal-20260828-pages-7d47b03/config}"
VKS_LAUNCHD_LABEL="${VKS_LAUNCHD_LABEL:-com.vetclub.knowledge-rag}"
VKS_MIN_FREE_GIB="${VKS_MIN_FREE_GIB:-20}"
VKS_FREE_GIB_OVERRIDE="${VKS_FREE_GIB_OVERRIDE:-}"
VKS_VERIFY_TIMEOUT="${VKS_VERIFY_TIMEOUT:-60}"

SNAP="$VKS_ROOT/snapshot"; SNAP_NEW="$VKS_ROOT/snapshot.new"
mkdir -p "$VKS_ROOT" "$VKS_STAGING_DOCS"
LOG="$VKS_ROOT/sync.log"; RECEIPTS="$VKS_ROOT/receipts.jsonl"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(now)" "$*" >>"$LOG"; }
receipt() { printf '%s\n' "$1" >>"$RECEIPTS"; printf '%s\n' "$1"; }

# 1. lock (mkdir-лок; снят trap-ом на EXIT)
if ! mkdir "$VKS_ROOT/lock" 2>/dev/null; then
  receipt '{"skip":"locked"}'; exit 0
fi
trap 'rm -rf "$VKS_ROOT/lock"' EXIT
log "lock: acquired"

# секрет: owner-only 0600; отказ с причиной до любого экспорта
if [ ! -f "$VKS_SECRET_ENV" ]; then
  log "secret: отсутствует $VKS_SECRET_ENV"; receipt '{"error":"secret_env"}'; exit 1
fi
SECRET_PERM=$(stat -f %Lp "$VKS_SECRET_ENV")
if [ "$SECRET_PERM" != "600" ]; then
  log "secret: права $SECRET_PERM != 600 на $VKS_SECRET_ENV"; receipt '{"error":"secret_env"}'; exit 1
fi

# 2. export (токен живёт только внутри подпроцесса)
T0=$(date +%s)
rm -rf "$SNAP_NEW"; mkdir -p "$SNAP_NEW"
if ! (set +u; set -a; . "$VKS_SECRET_ENV"; set +a; exec $VKS_EXPORTER --out "$SNAP_NEW") >>"$LOG" 2>&1; then
  log "export: ошибка"; receipt '{"error":"export"}'; exit 1
fi
T_EXPORT=$(date +%s); log "export: ok"

# 3. normalize + digest: без строк времени экспорта и провенанса
snapshot_digest() {
  { find "$1" -maxdepth 1 -type f -name '*.json' | LC_ALL=C sort
    find "$1/pages" -type f -name '*.md' 2>/dev/null | LC_ALL=C sort; } \
  | while IFS= read -r f; do
      grep -vE '^(\"exported_at\"|"source_commit"|exported_at:|source_commit:)' "$f" || true
    done | shasum -a 256 | awk '{print $1}'
}
DIG_NEW=$(snapshot_digest "$SNAP_NEW")
if [ -d "$SNAP" ]; then DIG_OLD=$(snapshot_digest "$SNAP"); else DIG_OLD=""; fi
log "digest: new=$DIG_NEW old=${DIG_OLD:-<нет>}"

# 4. diff
if [ -n "$DIG_OLD" ] && [ "$DIG_OLD" = "$DIG_NEW" ]; then
  receipt "{\"unchanged\":true,\"digest\":\"$DIG_NEW\"}"
  rm -rf "$SNAP_NEW"; log "diff: без изменений"; exit 0
fi

# 5. gate: свободное место на / (ГиБ)
if [ -n "$VKS_FREE_GIB_OVERRIDE" ]; then FREE_GIB="$VKS_FREE_GIB_OVERRIDE"
else FREE_GIB=$(df -g / | awk 'NR==2 {print $4}'); fi
if [ "${FREE_GIB%.*}" -lt "$VKS_MIN_FREE_GIB" ]; then
  receipt "{\"skip\":\"disk_low\",\"free_gib\":$FREE_GIB}"
  log "gate: disk_low free=${FREE_GIB}GiB, снимок не принят"; exit 0
fi
log "gate: free=${FREE_GIB}GiB >= $VKS_MIN_FREE_GIB"

# 6. bridge → временный корпус → замена notion-vet + staged-manifest
DIG12=${DIG_NEW:0:12}; TMP_CORPUS="$VKS_ROOT/notion-vet.tmp"
rm -rf "$TMP_CORPUS"
if ! $VKS_BRIDGE --snapshot-dir "$SNAP_NEW" --out "$TMP_CORPUS" --source-commit "sync-$DIG12" >>"$LOG" 2>&1; then
  log "bridge: ошибка"; receipt '{"error":"bridge"}'; exit 1
fi
if [ -d "$VKS_STAGING_DOCS/notion-vet" ]; then
  rm -rf "$VKS_ROOT/notion-vet.prev"
  mv "$VKS_STAGING_DOCS/notion-vet" "$VKS_ROOT/notion-vet.prev"
fi
mv "$TMP_CORPUS" "$VKS_STAGING_DOCS/notion-vet"
( cd "$VKS_STAGING_DOCS" && find . -type f | LC_ALL=C sort | xargs shasum -a 256 ) \
  > "$VKS_STAGING_DOCS/../staged-manifest.sha256"
T_BRIDGE=$(date +%s); log "bridge: ok sync-$DIG12, manifest пересобран"

# 7. build
GEN_ID="gen-sync-$(date -u +%Y%m%dT%H%MZ)-$DIG12"
if ! KNOWLEDGE_RAG_DIR="$KNOWLEDGE_RAG_DIR" "$VKS_GEN_CLI" build --generation-id "$GEN_ID" >>"$LOG" 2>&1; then
  log "build: ошибка $GEN_ID"; receipt '{"error":"build"}'; exit 1
fi
T_BUILD=$(date +%s); log "build: ok $GEN_ID"

# 8. restart + verify: servable=true и указатель на построенное поколение
launchctl kickstart -k "gui/$(id -u)/$VKS_LAUNCHD_LABEL" >>"$LOG" 2>&1
WAITED=0; SERVED=""
while [ "$WAITED" -lt "$VKS_VERIFY_TIMEOUT" ]; do
  ST=$("$VKS_GEN_CLI" status 2>>"$LOG" || true)
  if printf '%s' "$ST" | grep -Eq '"servable": *true' && printf '%s' "$ST" | grep -q "$GEN_ID"; then
    SERVED=1; break
  fi
  sleep 2; WAITED=$((WAITED + 2))
done
if [ -z "$SERVED" ]; then
  log "verify: сервер не на $GEN_ID"; receipt '{"error":"server_pointer"}'; exit 1
fi
log "verify: servable, generation $GEN_ID"

# 9. accept: снимок становится принятым, итоговая квитанция
rm -rf "$SNAP"; mv "$SNAP_NEW" "$SNAP"
receipt "{\"ts\":\"$(now)\",\"digest_before\":\"$DIG_OLD\",\"digest_after\":\"$DIG_NEW\",\"generation_id\":\"$GEN_ID\",\"seconds\":{\"export\":$((T_EXPORT - T0)),\"bridge\":$((T_BRIDGE - T_EXPORT)),\"build\":$((T_BUILD - T_BRIDGE))}}"
log "accept: $GEN_ID"
