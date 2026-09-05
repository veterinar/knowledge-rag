#!/bin/bash
# Установка/удаление/статус LaunchAgent ежечасной синхронизации Notion
# (AC-S10, контракт docs/criteria-notion-hourly-sync.md). Установка НЕ
# выполняется автоматически: только явный `install` владельцем.
set -euo pipefail

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
        sed -e "s|__SCRIPT__|$SCRIPT|" \
            -e "s|__LOGDIR__|$LOGDIR|" \
            -e "s|__HOME__|$HOME|" \
            "$TEMPLATE" > "$PLIST"
        launchctl bootstrap "gui/$UID_" "$PLIST"
        echo "install: $PLIST"
        launchctl print "gui/$UID_/$LABEL" | head
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
