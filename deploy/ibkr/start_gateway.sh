#!/bin/bash
# Start IB Gateway (paper, port 4002) through IBC on a virtual display, unless it is running.
# Run at boot and every 5 minutes from crontab; safe to run repeatedly.
export DISPLAY=:99
LOG=$HOME/ibkr/start_gateway.log
if ! pgrep -x Xvfb >/dev/null; then
    nohup Xvfb :99 -screen 0 1024x768x24 -fbdir $HOME/ibkr/xvfb >/dev/null 2>&1 &
    sleep 2
fi
if [ -f $HOME/ibkr/PAUSED ]; then
    exit 0   # paused (e.g. after a refused login): remove ~/ibkr/PAUSED to resume
fi
if grep -q "PUT_YOUR_PAPER" $HOME/ibc/config.ini; then
    echo "$(date '+%F %T') not started: enter the paper username/password in ~/ibc/config.ini" >> $LOG
    exit 0
fi
if ! pgrep -f "java.*ibcalpha.ibc.IbcGateway" >/dev/null; then
    echo "$(date '+%F %T') starting IB Gateway via IBC" >> $LOG
    nohup $HOME/ibc/gatewaystart.sh -inline >> $LOG 2>&1 &
fi
