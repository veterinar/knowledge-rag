#!/bin/bash
# Установка/удаление/статус LaunchAgent ежечасной синхронизации Notion
# (AC-S10, контракт docs/criteria-notion-hourly-sync.md). Установка НЕ
# выполняется автоматически: только явный `install` владельцем.
set -euo pipefail
umask 077

LABEL="com.vetclub.notion-knowledge-sync"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TEMPLATE="$SCRIPT_DIR/launchd/$LABEL.plist.template"
SCRIPT="$SCRIPT_DIR/notion_knowledge_sync.sh"
VKS_ROOT="${VKS_ROOT:-$HOME/.local/share/vetclub-knowledge-rag/sync}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_=$(id -u)

case "${1:-}" in
    install)
        LOGDIR="$VKS_ROOT/logs"
        mkdir -p "$LOGDIR" "$HOME/Library/LaunchAgents"
        # Подстановка плейсхолдеров через python: безопасно для & и | в путях
        /usr/bin/python3 - "$TEMPLATE" "$PLIST" "$SCRIPT" "$LOGDIR" "$HOME" <<'PY'
import sys
template, plist, script, logdir, home = sys.argv[1:6]
data = open(template, encoding="utf-8").read()
for key, value in (
    ("__SCRIPT__", script),
    ("__LOGDIR__", logdir),
    ("__HOME__", home),
):
    data = data.replace(key, value)
open(plist, "w", encoding="utf-8").write(data)
PY
        plutil -lint "$PLIST"
        # идемпотентность: выгрузить прошлую версию, если была
        launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
        launchctl bootstrap "gui/$UID_" "$PLIST"
        echo "install: $PLIST"
        launchctl print "gui/$UID_/$LABEL" > "$LOGDIR/install-print.txt" 2>&1 || true
        ;;
    uninstall)
        launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
        rm -f "$PLIST"
        echo "uninstall: $PLIST удалён, агент выгружен"
        ;;
    status)
        launchctl print "gui/$UID_/$LABEL"
        ;;
    *)
        echo "usage: $0 install|uninstall|status" >&2
        exit 2
        ;;
esac
