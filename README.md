# SwarmCore (рабочее название)

Плата с ПО, которая подключается к полётному контроллеру серийного беспилотника и
позволяет одному оператору вести группу аппаратов: делить работу, разводить их в воздухе
и продолжать миссию, если один аппарат выбыл. Применение — гражданское (агрообработка,
аэрофотосъёмка, мониторинг). Покупатель — производители БВС.

Проект готовится к акселератору МАИ (подача заявки до 20 ноября 2026).

## Документы

Начинать с **[docs/00_summary.md](docs/00_summary.md)** — резюме проекта на 2 страницы.

| Файл | Что внутри |
|---|---|
| [00_summary.md](docs/00_summary.md) | резюме: проблема, решение, клиент, MVP, бизнес-модель |
| [01_problem_and_customer.md](docs/01_problem_and_customer.md) | проблема, сегменты, гипотезы ценности |
| [02_product.md](docs/02_product.md) | продукт: состав платы, функции по версиям, чего не делаем |
| [03_market_and_competitors.md](docs/03_market_and_competitors.md) | рынок, оценка объёма, конкуренты |
| [04_business_model.md](docs/04_business_model.md) | монетизация, цены, юнит-экономика |
| [05_mvp_and_plan.md](docs/05_mvp_and_plan.md) | MVP к 20 ноября и план на полгода |
| [06_custdev.md](docs/06_custdev.md) | интервью с клиентами: кого, о чём, пороги |
| [07_team.md](docs/07_team.md) | команда и роли |
| [08_risks.md](docs/08_risks.md) | риски |
| [09_tech_appendix.md](docs/09_tech_appendix.md) | техническое приложение |
| [archive/](docs/archive/) | прошлая исследовательская версия документации |

## Код

* `swarmsim/` — алгоритмы (деление участка `sectors.py`, распределение и перераспределение
  `alloc.py`, маршрут и валидатор `plan.py`) и симулятор со стендом прогонов. Описание —
  [swarmsim/README.md](swarmsim/README.md) и [docs/m2/README.md](docs/m2/README.md).
* `swarmcore_px4/` — адаптер к PX4 через MAVSDK и запуск нескольких PX4 SITL.
* `apps/` — командный интерфейс `swarmcore` и визуализация.

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev,viz]"
./.venv/bin/python -m pytest -q                       # все тесты (SITL-тест пропускается)
./.venv/bin/swarmcore run --scenario scenarios/failures.yaml --seed 1 --set allocator.name=central_greedy
./.venv/bin/python scripts/report.py --alloc          # время расчёта и баланс загрузки
./.venv/bin/python scripts/visualize.py --plan        # картинка плана
```

`scripts/create_issues.sh` создаёт задачи по **старому** плану (P0–P6) — не запускать;
новая доска описана в [docs/05_mvp_and_plan.md](docs/05_mvp_and_plan.md).
