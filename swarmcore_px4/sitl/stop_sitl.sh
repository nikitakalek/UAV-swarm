#!/usr/bin/env bash
# Остановить все экземпляры PX4 SITL и Gazebo, запущенные multi_sitl.sh.
RUN_DIR=${RUN_DIR:-/tmp/swarmcore_sitl}
if [[ -f "$RUN_DIR/pids" ]]; then
  while read -r pid; do kill "$pid" 2>/dev/null; done < "$RUN_DIR/pids"
  rm -f "$RUN_DIR/pids"
fi
sleep 1
pkill -f "[p]x4_sitl_default/bin/px4" 2>/dev/null
pkill -f "[g]z sim" 2>/dev/null
pkill -f "[g]z-sim" 2>/dev/null
pkill -f "[m]avsdk_server" 2>/dev/null
sleep 2
pkill -9 -f "[p]x4_sitl_default/bin/px4" 2>/dev/null
pkill -9 -f "[g]z sim" 2>/dev/null
pkill -9 -f "[m]avsdk_server" 2>/dev/null
echo "SITL остановлен"
