#!/usr/bin/env bash
# M2-12: запуск N экземпляров PX4 SITL + Gazebo (Harmonic) одной командой.
#
#   ./multi_sitl.sh 3                # 3 аппарата, без графики
#   GUI=1 ./multi_sitl.sh 2          # с окном Gazebo
#   HOMES="0,0 4,0 8,0" ./multi_sitl.sh 3   # точки старта (x восток, y север, м)
#
# Экземпляр i получает:  MAV_SYS_ID = i+1,  порт offboard API (MAVSDK) = 14540+i,
# порт наземной станции = 14550, свою папку build/px4_sitl_default/instance_i.
# Первый экземпляр поднимает сервер Gazebo, остальные подключаются к нему
# (PX4_GZ_STANDALONE=1). Логи и PID — в $RUN_DIR (по умолчанию /tmp/swarmcore_sitl).
set -euo pipefail

N=${1:-2}
PX4_DIR=${PX4_DIR:-$HOME/PX4-Autopilot}
MODEL=${MODEL:-gz_x500}
AUTOSTART=${AUTOSTART:-4001}
RUN_DIR=${RUN_DIR:-/tmp/swarmcore_sitl}
WORLD=${WORLD:-default}
read -r -a HOME_ARR <<< "${HOMES:-}"

if [[ ! -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]]; then
  echo "PX4 не собран: $PX4_DIR/build/px4_sitl_default/bin/px4 (make px4_sitl)" >&2
  exit 1
fi
if [[ -f "$RUN_DIR/pids" ]]; then
  echo "похоже, SITL уже запущен ($RUN_DIR/pids); сначала stop_sitl.sh" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
: > "$RUN_DIR/pids"
: > "$RUN_DIR/instances.csv"
echo "instance,sysid,mavsdk_port,home_x,home_y" >> "$RUN_DIR/instances.csv"

cd "$PX4_DIR"
for ((i = 0; i < N; i++)); do
  pose=${HOME_ARR[$i]:-"$((i * 4)),0"}
  hx=${pose%,*}; hy=${pose#*,}
  # PX4 проверяет PX4_GZ_STANDALONE на «задана» (-z), поэтому первому экземпляру её не передаём
  standalone=
  [[ $i -gt 0 ]] && standalone=1
  headless=1
  [[ "${GUI:-0}" == "1" && $i -eq 0 ]] && headless=
  env PX4_SYS_AUTOSTART=$AUTOSTART PX4_SIM_MODEL=$MODEL PX4_GZ_MODEL_POSE="$hx,$hy" \
      ${standalone:+PX4_GZ_STANDALONE=1} PX4_GZ_WORLD=$WORLD ${headless:+HEADLESS=1} \
      nohup ./build/px4_sitl_default/bin/px4 -i "$i" -d > "$RUN_DIR/px4_$i.log" 2>&1 &
  echo $! >> "$RUN_DIR/pids"
  echo "$i,$((i + 1)),$((14540 + i)),$hx,$hy" >> "$RUN_DIR/instances.csv"
  echo "PX4 #$i: sysid=$((i + 1)) порт MAVSDK udp $((14540 + i)) старт ($hx,$hy) лог $RUN_DIR/px4_$i.log"
  if [[ $i -eq 0 ]]; then
    # ждём, пока Gazebo поднимет мир, иначе standalone-экземпляры не найдут сервер
    for _ in $(seq 1 60); do
      gz topic -l 2>/dev/null | grep -q "/world/$WORLD/clock" && break
      sleep 1
    done
  fi
  # следующий экземпляр — когда текущий заспавнил модель
  for _ in $(seq 1 60); do
    grep -q "Ready for takeoff\|home set\|INFO  \[commander\] Ready" "$RUN_DIR/px4_$i.log" 2>/dev/null && break
    grep -q "ERROR \[gz_bridge\]" "$RUN_DIR/px4_$i.log" 2>/dev/null && { echo "ошибка gz_bridge, см. лог" >&2; break; }
    sleep 1
  done
done
echo "запущено $N экземпляров; остановка: $(dirname "$0")/stop_sitl.sh"
