# Модуль М2 — симулятор и испытательный стенд

Владелец М2: ядро симулятора, среда, детерминизм, стенд прогонов, командный
интерфейс, сценарии, журнал событий, визуализация, адаптер PX4.

## Быстрый старт

```bash
pip install -e ".[dev,viz]"
pytest -q
swarmcore run --scenario scenarios/nominal.yaml --seed 1
swarmcore run --scenario scenarios/failures.yaml --seeds 1-20 --jobs 4
swarmcore compare runs/nominal__stub_roundrobin runs/failures__stub_roundrobin
swarmcore explain runs/failures__stub_roundrobin/seed_001
swarmcore viz runs/failures__stub_roundrobin/seed_001 --out failures.gif --png failures.png
```

Результат прогона (`runs/<сценарий>__<распределитель>/seed_NNN/`): `metrics.csv/json`,
`trajectory.csv`, `tasks.csv`, `events.jsonl`, `world.json`, `scenario.json`,
`run_manifest.json` (хэш конфига, зерно, хэши метрик и журнала, версия, коммит).
Для пакета зёрен — ещё `summary.csv` и `batch_manifest.json`.

## Кому какие файлы принадлежат

| Файл | Владелец |
|---|---|
| `swarmsim/world.py`, `geometry.py`, `sim.py`, `scenario.py`, `eventlog.py`, `registry.py` | М2 |
| `swarmsim/agents/` (`MotionSource`, `Agent`, `KinematicMotion`) | М2 |
| `swarmsim/missions/`, `swarmsim/planning/coverage.py` | М2 |
| `apps/cli.py`, `apps/viz.py`, `swarmcore_px4/`, `scenarios/`, CI | М2 |
| `swarmsim/interfaces.py` | вся команда (стыки GEN-2) |
| `swarmsim/stubs.py`, `swarmsim/basic_metrics.py` | **временные заглушки**, заменяются М1/М3/INT |
| `swarmsim/alloc/`, `planning/partition.py`, `planning/routing.py`, `planning/validator.py`, `swarmcore_node/` | М1 |
| `sensing.py`, `comms.py`, `faults.py`, энергия | М3 (не создавались) |
| `metrics.py`, `experiments/` | интеграция (не создавались) |

## Как подключить свой модуль (М1/М3) без правки ядра

Реализуйте протокол из `swarmsim/interfaces.py` и укажите класс в сценарии:

```yaml
allocator: {name: "swarmsim.alloc.cbba:CBBAAllocator", params: {...}}
sensor:    {model: "swarmsim.sensing:CameraSensor", swath_width: 40}
comms:     {model: "swarmsim.comms:CommsBus", delay: 0.2, loss: 0.1}
faults:    {model: "swarmsim.faults:FaultManager", detect_delay: 2.0}
energy:    {model: "swarmsim.energy:ZengEnergy"}
```

или зарегистрируйте короткое имя: `registry.register_allocator("cbba", "swarmsim.alloc.cbba:CBBAAllocator")`.
Если конструктор принимает `rng` — симулятор передаст ему собственный поток случайных
чисел (детерминизм по зерну); если `scenario` — весь сценарий.

Необязательные методы распределителя: `should_trigger(t, tasks, pending)` (только модельное
время и события!), `requires_gcs` (по умолчанию `True`), `on_message(msg)`.

## Статус задач М2

| Задача | Что сделано | Проверка |
|---|---|---|
| M2-1 Каркас и CI | пакет `swarmsim/`, `pyproject.toml`, `.github/workflows/ci.yml` (pytest + повторный прогон с `compare --strict`) | `pip install -e .`, `pytest` |
| M2-2 World | границы (любой полигон), бесполётные зоны, цели, карта вероятности (равномерная/кластеры), `step`, `observe`, обход зон | `tests/test_world.py` |
| M2-3 KinematicMotion | безмассовая модель, перенос остатка пути между точками, `v ≤ v_max`, интерфейс `MotionSource` | `tests/test_kinematics.py` |
| M2-4 SimLoop | `run(scenario, seed)`, `t = k·dt`, отдельные потоки ГСЧ, `run_manifest.json`, независимость от настенного времени | `tests/test_sim.py` |
| M2-5 MissionPlugin | сетка N ячеек + «змейка», покрытие 100 % (квадрат, L-область, треугольник) | `tests/test_mission.py` |
| M2-6 Стенд и CLI | `swarmcore run/compare/validate/viz/explain`, 20 зёрен одной командой, `--jobs`, `--set` | `tests/test_cli.py` |
| M2-7 Сценарии | pydantic-схема (`schema_version: 1`), события `agent_failure/new_target/comms_loss`, случайные отказы по зерну, 7 сценариев | `tests/test_scenarios.py` |
| M2-8 Журнал | `EventLog`, цепочка причин, `explain`, формат в `eventlog_format.md` | `tests/test_eventlog.py` |
| M2-9 Неидеальный канал | доставка по `(t_deliver, src, seq)`, тест с задержкой/дрожанием/потерями | `tests/test_sim.py::test_lossy_channel_*` |
| M2-10 Визуализация | анимация GIF/MP4 и сводный PNG только из файлов прогона | `swarmcore viz` |
| M2-11 MavsdkMotion | тот же `MotionSource` поверх MAVSDK (offboard, уставки скорости), замер задержки | `python -m swarmcore_px4.run_sitl latency` |
| M2-12 Мульти-SITL | `swarmcore_px4/sitl/multi_sitl.sh N`: `MAV_SYS_ID=i+1`, порт `14540+i` | `python -m swarmcore_px4.run_sitl check --n 2` |
| M2-13 Ручные сектора | `mission.sectors` (версия 1), валидация (пересечения, покрытие, бесполётные зоны), режимы `centroid/overlap/clip`, распределитель `manual_sectors` | `tests/test_sectors.py`, `scenarios/manual_sectors.yaml` |

## Известные ограничения

* Заглушки М1/М3 намеренно примитивны: `stub_roundrobin` ставит освобождённые задачи в
  конец очереди, поэтому `T_recovery` у него большой — это свойство заглушки, а не ядра.
* Галс «змейки» может выходить за **наклонную** границу области не более чем на `W/2`
  (цена гарантии покрытия 100 %). У бесполётных зон галсы обрезаются с запасом, поэтому
  у их краёв покрытие чуть меньше 100 % (тест требует > 99 %).
* Обход бесполётных зон — граф видимости по углам их рамок (не оптимальный путь).
* Столкновения считаются в 3D; без тактического избегания (M1-14) безопасность
  обеспечивается эшелонированием по высоте (`agents.altitude_step`).
* SITL-прогон идёт в реальном времени, поэтому побитовой воспроизводимости там нет
  (`mode: sitl` в manifest).
