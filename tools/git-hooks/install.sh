#!/bin/sh
# Install the forward-only git hooks (prepare-commit-msg, pre-rebase,
# pre-push, reference-transaction) into a repository's hooks directory.
#
#   tools/git-hooks/install.sh [--all-worktrees] [--refuse-all-push] [repo ...]
#
# repo defaults to the current directory.  Worktrees share their main
# repository's hooks directory, so one install covers them all; with
# --all-worktrees every worktree is still checked for its own
# core.hooksPath (extensions.worktreeConfig) and each distinct hooks
# directory gets the hooks.  core.hooksPath, when set, is honoured.
# --refuse-all-push sets forwardonly.refuseAllPush=true, for a
# repository that never pushes anywhere.
#
# A hook already present that is not ours is kept and chained: it is
# renamed <name>.local and our hook runs it after its own check.
# Re-running the installer refreshes our hooks in place.

set -u
here=$(cd "$(dirname "$0")" && pwd)
HOOKS="prepare-commit-msg pre-rebase pre-push reference-transaction"
MARK="forward-only-hooks:"
all=0
nopush=0
repos=""
for a in "$@"; do
    case "$a" in
        --all-worktrees) all=1 ;;
        --refuse-all-push) nopush=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) repos="$repos
$a" ;;
    esac
done
[ -n "$repos" ] || repos=.

norm() {
    # One spelling per directory (C:/x and /c/x are the same on Windows).
    (cd "$1" 2>/dev/null && pwd -P) || printf '%s\n' "$1"
}

hooks_dir_of() {
    # $1: a worktree or repository directory.  Prints its absolute
    # hooks directory.
    hp=$(git -C "$1" config --get core.hooksPath 2>/dev/null || true)
    if [ -n "$hp" ]; then
        case "$hp" in
            "~"/*) hp="$HOME/${hp#"~/"}" ;;
        esac
        case "$hp" in
            /*|[A-Za-z]:*) ;;
            *)  top=$(git -C "$1" rev-parse --show-toplevel 2>/dev/null || git -C "$1" rev-parse --absolute-git-dir)
                hp="$top/$hp" ;;
        esac
        norm "$hp"
    else
        common=$(git -C "$1" rev-parse --git-common-dir) || return 1
        case "$common" in
            /*|[A-Za-z]:*) ;;
            *) common="$(cd "$1" && cd "$common" && pwd)" ;;
        esac
        norm "$common/hooks"
    fi
}

install_into() {
    d=$1
    mkdir -p "$d" || return 1
    for h in $HOOKS; do
        t="$d/$h"
        if [ -e "$t" ] && ! grep -q "$MARK" "$t" 2>/dev/null; then
            if [ -e "$t.local" ]; then
                echo "skip $t: an unrecognised hook and $t.local both exist; merge them by hand" >&2
                continue
            fi
            mv "$t" "$t.local"
            echo "kept the existing $h as $t.local (chained)"
        fi
        cp "$here/$h" "$t.tmp.$$" && chmod 755 "$t.tmp.$$" && mv -f "$t.tmp.$$" "$t" || return 1
    done
    echo "installed: $d"
}

printf '%s\n' "$repos" | while IFS= read -r r; do
    [ -n "$r" ] || continue
    if ! git -C "$r" rev-parse --git-dir >/dev/null 2>&1; then
        echo "not a git repository: $r" >&2
        exit 1
    fi
    if [ "$nopush" = 1 ]; then
        git -C "$r" config forwardonly.refuseAllPush true
    fi
    if [ "$all" = 1 ]; then
        wts=$(git -C "$r" worktree list --porcelain | sed -n 's/^worktree //p')
    else
        wts=$r
    fi
    printf '%s\n' "$wts" | while IFS= read -r w; do
        [ -d "$w" ] || continue
        d=$(hooks_dir_of "$w") || continue
        printf '%s\n' "$d"
    done
done | sort -u | while IFS= read -r d; do
    install_into "$d" || exit 1
done
