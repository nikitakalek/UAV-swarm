# Формат журнала событий (M2-8), версия 1

Файл `events.jsonl` в папке прогона. Первая строка — заголовок:

```json
{"format": "swarmcore-eventlog", "version": 1}
```

Далее — по одной записи на строку:

| Поле | Тип | Смысл |
|---|---|---|
| `seq` | int | порядковый номер (0, 1, 2, …), уникален в прогоне |
| `t` | float | модельное время, с |
| `type` | str | тип события (см. ниже) |
| `source` | str | кто породил: `scenario`, `sim`, `faults`, `mission`, `allocator:<имя>`, `agent:<id>` |
| `cause` | int \| null | `seq` события-причины; всегда ссылается на более раннюю запись |
| `data` | object | данные события |

## Типы событий

| type | source | data | cause |
|---|---|---|---|
| `mission_start` | sim | `n_agents`, `n_tasks`, `allocator`, `seed` | — (корень) |
| `agent_failure` | scenario | `agent`, `pos`, `random?` | — (корень) |
| `failure_detected` | faults | `agent` | `agent_failure` |
| `task_released` | sim | `task`, `from`, `to=null`, `reason` | `failure_detected` |
| `new_target` | scenario | `target`, `pos` | — (корень) |
| `task_created` | mission | `task`, `reason` | `new_target` |
| `comms_loss` | scenario | `scope` (`gcs`/`all`), `until` | — (корень) |
| `comms_restored` | scenario | `scope`, `loss_seq` | — (корень) |
| `sim_trigger` | sim | `reason` | — (появились неназначенные задачи без события) |
| `allocation` | allocator:… | `trigger` (тип события-причины) | событие, вызвавшее перераспределение |
| `assign` / `reassign` / `unassign` | allocator:… | `task`, `from`, `to`, `reason` | `allocation` |
| `allocation_warning` | sim | `reason`, `task?` | `allocation` |
| `task_done` | agent:… | `task` | — |
| `target_detected` | agent:… | `target` | — |
| `mission_end` | sim | `done`, `total` | — |

## Как восстановить причину переназначения

Идём по `cause` от записи `assign`/`reassign` до корня:

```
assign cell_r02_c03 None→6  →  allocation  →  failure_detected (A3)  →  agent_failure (A3, t=150)
```

В коде: `EventLog.chain(seq)`, `EventLog.root_cause(seq)`, `EventLog.explain_task(task_id)`.
Из командной строки: `swarmcore explain runs/<группа>/seed_001 [--task ID]`.

`reason` даёт распределитель (поле `Assignment.reasons`); если он его не дал,
симулятор пишет тип события-причины.

## Совместимость

Новые типы событий и новые поля в `data` можно добавлять без смены версии.
Переименование или удаление полей требует новой `version` в заголовке.
