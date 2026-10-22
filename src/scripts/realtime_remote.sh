#!/usr/bin/env bash
# Run on the Mac: start (if needed) the preview server on a remote host in
# tmux, tunnel it over SSH and open it in the local browser. Ctrl-C closes
# the tunnel; the server keeps running for the next connection.
#
#   bash src/scripts/realtime_remote.sh                 # NYUAIRLAB-127
#   bash src/scripts/realtime_remote.sh --restart       # pull code + restart server
#   bash src/scripts/realtime_remote.sh --stop          # stop the remote server
#
# Env: HOST (ssh alias), PORT (remote port), REMOTE_DIR (repo on the host).
set -euo pipefail
HOST=${HOST:-NYUAIRLAB-127}
PORT=${PORT:-8765}
REMOTE_DIR=${REMOTE_DIR:-'~/langtian/BreastSIM'}
SESSION=breastsim_rt
ACTION=${1:-start}

remote() { ssh -o BatchMode=yes "$HOST" "$@"; }

case "$ACTION" in
    --stop) remote "tmux kill-session -t $SESSION 2>/dev/null && echo stopped || echo 'not running'"; exit 0 ;;
    --restart) remote "tmux kill-session -t $SESSION 2>/dev/null || true" ;;
esac

echo "[remote] $HOST: syncing code and starting server (tmux $SESSION, port $PORT)"
remote "cd $REMOTE_DIR && (git pull --ff-only -q || echo 'git pull skipped') && \
    (tmux has-session -t $SESSION 2>/dev/null || tmux new-session -d -s $SESSION 'PORT=$PORT bash src/scripts/realtime_server.sh')"
remote "for i in \$(seq 90); do curl -sf http://127.0.0.1:$PORT/api/params >/dev/null && exit 0; sleep 1; done; \
    echo 'server did not come up; tail of log:'; tail -20 $REMOTE_DIR/logs/realtime.log; exit 1"

LOCAL=$PORT
while lsof -iTCP:"$LOCAL" -sTCP:LISTEN >/dev/null 2>&1; do LOCAL=$((LOCAL + 1)); done
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L "$LOCAL:127.0.0.1:$PORT" "$HOST" &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null; echo; echo "[remote] tunnel closed (server still running on $HOST; --stop to stop it)"' EXIT
sleep 1
URL="http://localhost:$LOCAL"
echo "[remote] tunnel up: $URL  (Ctrl-C to close)"
command -v open >/dev/null && open "$URL"
wait $TUNNEL
