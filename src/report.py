"""
Сводный отчёт: прогоняет Greedy, CpSat и Model на одних данных и строит сравнительную таблицу.
"""

import json
import subprocess
import sys
import tempfile
import time as time_module
from pathlib import Path

import pandas as pd

from greedy import Greedy
from prepare import Prepare

_SRC_DIR = str(Path(__file__).resolve().parent)


class Report:
    """Прогоняет три подхода к назначению техники на одних данных и сравнивает результат."""

    def __init__(self, prepare: Prepare):
        self.prepare = prepare
        self.data = prepare.data

    def run(self, time_limit_sec: float = 300.0, mip_rel_gap: float = 0.005) -> "Report":
        """
            Запускает Greedy (в этом же процессе), затем CpSat и Model (каждый
            — в отдельном чистом процессе, см. docstring модуля) и строит
            self.summary / self.table. Исходный self.prepare не меняется —
            каждый солвер загружает свои Data/Prepare заново.
        """

        t0 = time_module.time()
        greedy = Greedy(self.prepare).run()
        greedy_time = time_module.time() - t0

        cpsat_assignment, cpsat_meta = self._run_isolated("cpsat", "CpSat", time_limit_sec, mip_rel_gap)
        mip_assignment, mip_meta = self._run_isolated("mip", "Model", time_limit_sec, mip_rel_gap)

        self.results = {
            "Greedy": (greedy.assignment, "—", greedy_time),
            "CP-SAT": (cpsat_assignment, cpsat_meta["status"], cpsat_meta["solve_time_sec"]),
            "MIP (HiGHS)": (mip_assignment, mip_meta["status"], mip_meta["solve_time_sec"]),
        }

        self.summary = self._build_summary()
        self.table = self._build_table()

        return self

    def _run_isolated(self, module_name: str, class_name: str, time_limit_sec: float,
                       mip_rel_gap: float) -> tuple:
        """
            Решает {module_name}.{class_name} в отдельном процессе: solver
            сам загружает Data/Prepare (дёшево, доли секунды — обмен через
            файл проще и надёжнее, чем сериализовать Prepare с "живыми"
            переменными солвера внутри). Возвращает (assignment, meta), где
            meta — словарь {status, solve_time_sec} из последней строки stdout.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "assignment.csv"
            code = f"""
import sys, json
sys.path.insert(0, {_SRC_DIR!r})
from data import Data
from prepare import Prepare
from {module_name} import {class_name}

solver = {class_name}(Prepare(Data()), time_limit_sec={time_limit_sec!r}, mip_rel_gap={mip_rel_gap!r})
solver.solve()
solver.assignment.to_csv({str(out_path)!r}, index=False)
print(json.dumps({{"status": solver.status, "solve_time_sec": solver.solve_time_sec}}), flush=True)
"""
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
            meta = json.loads(result.stdout.strip().splitlines()[-1])
            assignment = pd.read_csv(out_path)

        return assignment, meta

    def _build_summary(self) -> pd.DataFrame:
        """Одна строка на подход: заказы (всего/обязательные), затраты на перегон, время решения."""

        rows = [self._summarize_one(label, *data) for label, data in self.results.items()]

        return pd.DataFrame(rows).set_index("approach")

    def _summarize_one(self, label: str, assignment: pd.DataFrame, status: str, solve_time_sec: float) -> dict:
        """Метрики одного подхода по его итоговому assignment (equipment_id, order_id, leg_id, start_time_h)."""

        orders = self.data.orders

        total_orders = orders["order_id"].nunique()
        fulfilled_orders = assignment["order_id"].nunique()

        mandatory_orders = orders.loc[orders["is_mandatory"], "order_id"].unique()
        fulfilled_mandatory = assignment.loc[assignment["order_id"].isin(mandatory_orders), "order_id"].nunique()

        travel_cost = self._travel_cost(assignment)

        return {
            "approach": label,
            "orders_fulfilled": fulfilled_orders,
            "orders_total": total_orders,
            "fulfilled_pct": round(100 * fulfilled_orders / total_orders, 1),
            "mandatory_fulfilled": fulfilled_mandatory,
            "mandatory_total": len(mandatory_orders),
            "travel_cost_rub": round(travel_cost, 2),
            "cost_per_order_rub": round(travel_cost / fulfilled_orders, 2) if fulfilled_orders else None,
            "solve_time_sec": round(solve_time_sec, 2),
            "status": status,
        }

    def _travel_cost(self, assignment: pd.DataFrame) -> float:
        """
            Затраты на перегон по факту решения: для каждой единицы техники
            — от депо до первой её работы, плюс между последовательными
            работами в её хронологической цепочке. Считается заново по
            assignment одинаково для всех трёх подходов, а не берётся из
            внутренней бухгалтерии солвера — так расхождение сразу видно.
        """

        fleet = self.data.fleet.set_index("equipment_id")
        orders = self.data.orders.set_index("leg_id")
        cost = self.prepare.mobilization_cost

        total = 0.0
        for equipment_id, group in assignment.sort_values("start_time_h").groupby("equipment_id"):
            prev_location = fleet.loc[equipment_id, "depot_location"]
            for leg_id in group["leg_id"]:
                location = orders.loc[leg_id, "location"]
                total += float(cost.loc[prev_location, location])
                prev_location = location

        return total

    def _build_table(self) -> pd.DataFrame:
        """self.summary с русскими подписями метрик по строкам и подходами по столбцам — для печати/показа."""

        labels = {
            "orders_fulfilled": "Выполнено заказов, шт.",
            "fulfilled_pct": "Выполнено заказов, %",
            "mandatory_fulfilled": "из них обязательных, шт.",
            "travel_cost_rub": "Затраты на перегон, ₽",
            "cost_per_order_rub": "Затраты на 1 заказ, ₽",
            "solve_time_sec": "Время решения, с",
            "status": "Статус",
        }

        table = self.summary[list(labels)].T
        table.index = table.index.map(labels)

        return table

    def to_markdown(self) -> str:
        """self.table в виде markdown-таблицы (без зависимости от tabulate — таблица маленькая)."""

        table = self.table.reset_index().rename(columns={"index": "Метрика"})
        header = "| " + " | ".join(table.columns) + " |"
        separator = "|" + "|".join(["---"] * len(table.columns)) + "|"
        rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]

        return "\n".join([header, separator] + rows)
