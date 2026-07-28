#!/bin/bash
# Hook pre-push : empêche les pushs directs vers master ou main
# à partir d'une branche feature non mergée.
# Sauf si --no-verify est passé.

BRANCH=$(git symbolic-ref HEAD 2>/dev/null | sed 's/refs\/heads\///')
TARGET="$1"

if [ "$BRANCH" = "master" ] || [ "$BRANCH" = "main" ]; then
    echo "❌ Push direct sur $BRANCH interdit. Utilise une branche feature + merge."
    exit 1
fi

exit 0
