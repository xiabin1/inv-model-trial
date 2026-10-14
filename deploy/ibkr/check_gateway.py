"""IB Gateway health check (cron, trading days: 06:00, 08:40, every 30 min 09:00-15:30, 15:45 ET).

The watchdog (start_gateway.sh) only restarts the gateway when its process is gone. This check tests
what the orders need: an API connection on the configured port that sees the strategy's account. If that
fails (e.g. the gateway sits logged out after IBKR's weekly re-login, as on 2026-10-12), it restarts the
gateway through IBC, tests again, and sends a Telegram message: recovered automatically, or still down
(then the next 09:00 sells / 15:58 buys are at risk and need a manual login). It never trades.

    python deploy/ibkr/check_gateway.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.broker import ibkr  # noqa: E402
from src.notify import telegram  # noqa: E402

STATE = ROOT / "data/state/gateway_check.json"
PAUSED = Path.home() / "ibkr/PAUSED"
START = ROOT / "deploy/ibkr/start_gateway.sh"
WAIT_SECONDS = 180


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def connected() -> tuple[bool, str]:
    """(ok, detail): an API connection that sees the configured account."""
    config = ibkr.load_config()
    config["connection"]["timeout_seconds"] = 10
    config["connection"]["client_id"] = 91  # its own client id, so it never collides with a trading job
    try:
        ib = ibkr.connect(config)
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error)[:200]}"
    try:
        accounts = ib.managedAccounts()
    finally:
        ib.disconnect()
    return True, f"accounts {accounts}"


def restart() -> None:
    """Stop the gateway (by process id) and let the watchdog script start it again through IBC."""
    for pattern in ("java.*ibcalpha.ibc.IbcGateway", "ibcstart.sh"):
        found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
        for pid in found:
            subprocess.run(["kill", pid])
    time.sleep(8)
    subprocess.run([str(START)])


def remember(status: str) -> str:
    previous = json.loads(STATE.read_text()).get("status", "ok") if STATE.exists() else "ok"
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"status": status, "at": now()}))
    return previous


def main() -> int:
    if PAUSED.exists():
        print(f"{now()} gateway paused ({PAUSED}); not checked")
        if remember("paused") != "paused":
            telegram.alert(f"⏸ IBKR gateway is paused ({PAUSED} exists)\nNo orders can be sent until the file is removed.")
        return 0
    ok, detail = connected()
    if ok:
        print(f"{now()} gateway ok: {detail}")
        if remember("ok") != "ok":
            telegram.alert(f"✅ IBKR gateway is reachable again\n{detail}")
        return 0
    print(f"{now()} gateway NOT reachable: {detail}; restarting")
    restart()
    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        time.sleep(15)
        ok, after = connected()
        if ok:
            print(f"{now()} gateway ok after restart: {after}")
            remember("ok")
            telegram.alert(f"⚠️ IBKR gateway was down, restarted automatically and connected again\n"
                           f"was: {detail}\nnow: {after}")
            return 0
    print(f"{now()} gateway still down after restart: {after}")
    remember("down")
    telegram.alert("❌ IBKR gateway DOWN (automatic restart did not help)\n"
                   f"{after}\nThe next 09:00 sells / 15:58 buys will fail until it is logged in. "
                   "Check ~/ibc/logs (login refused? second factor? existing session?) and the IBKR credentials "
                   "in ~/ibc/config.ini; restart with deploy/ibkr/start_gateway.sh after stopping the gateway.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
