"""Та же модель назначения техники на заявки, но на CP-SAT (OR-Tools) вместо HiGHS."""

import time as time_module

import pandas as pd
from ortools.sat.python import cp_model

import config
from prepare import Prepare

_STATUS_NAMES = {
    cp_model.OPTIMAL: "Optimal",
    cp_model.FEASIBLE: "Feasible",
    cp_model.INFEASIBLE: "Infeasible",
    cp_model.UNKNOWN: "Unknown",
    cp_model.MODEL_INVALID: "ModelInvalid",
}

# Часы -> целые "сантичасы": CP-SAT работает только с целочисленными доменами.
TIME_SCALE = 100


class CpSat:
    """Строит и решает ТУ ЖЕ модель назначения, что Model (см. model.py), но на CP-SAT."""

    def __init__(self, prepare: Prepare, time_limit_sec: float = 300.0, mip_rel_gap: float = 0.005):
        self.prepare = prepare.clone()
        self.data = self.prepare.data
        self.time_limit_sec = time_limit_sec
        self.mip_rel_gap = mip_rel_gap

        self._model = cp_model.CpModel()
        self._start = None
        self._compiled = False

    def compile(self) -> "CpSat":
        """
            Строит переменные, ограничения и целевую функцию. Повторный вызов ничего не делает.
        """

        if self._compiled:
            return self

        # x[e,leg], u[e,leg], y[e,leg1,leg2], s_leg
        self._add_variables()
        # (1) не более одной единицы техники на ногу заявки
        self._add_one_equip_per_order_constraints()
        # (2) не более одной "первой работы" на единицу техники
        self._add_one_first_job_constraints()
        # подготовка списков входящих/исходящих переменных для (3)/(4)
        self._compute_flow_incidence()
        # (3) баланс потока: вход = назначение
        self._add_inflow_constraints()
        # (4) баланс потока: выход <= назначение
        self._add_outflow_constraints()
        # (5) работы должны завершиться до конца доступности техники
        self._add_availability_constraints()
        # (6) согласованность времени в цепочке
        self._add_chain_timing_constraints()
        # (7) инициализация времени начала первой заявки
        self._add_first_job_timing_constraints()
        # (8) комбинированные заказы: ноги с одинаковым order_id — вместе и одновременно
        self._add_combo_constraints()
        # избыточное, но полезное для CP-SAT: единица не может быть в двух местах разом
        self._add_no_overlap_constraints()
        # целевая функция: выполнение заказов минус перегон
        self._add_objective()

        self._compiled = True

        return self

    def solve(self) -> None:
        """
            Запускает решение (при необходимости сначала компилирует модель).
            После вызова доступны self.status, self.solve_time_sec,
            self.objective_value и self.assignment.
        """

        self.compile()

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.time_limit_sec)
        solver.parameters.relative_gap_limit = float(self.mip_rel_gap)
        solver.parameters.num_search_workers = 8
        solver.parameters.log_search_progress = True

        t0 = time_module.time()
        cp_status = solver.Solve(self._model)
        self.solve_time_sec = time_module.time() - t0

        self.status = _STATUS_NAMES.get(cp_status, str(cp_status))
        self.objective_value = solver.ObjectiveValue() if cp_status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
        self._solver = solver
        self.assignment = self._extract_assignment()

    def _scale(self, hours: float) -> int:
        """Часы -> целые "сантичасы" (TIME_SCALE), нужно для целочисленных доменов CP-SAT."""

        return int(round(hours * TIME_SCALE))

    def _add_variables(self) -> None:
        """
            Создаёт x[e,leg], u[e,leg], y[e,leg1,leg2] и s_leg. Переменные
            дописываются колонкой var прямо в таблицы Prepare — та же таблица
            дальше используется и в ограничениях, и в целевой функции.
        """

        model = self._model

        x = self.prepare.equipment_order_edges
        x["var"] = x.apply(lambda r: model.NewBoolVar(f"x_{r.equipment_id}_{r.leg_id}"), axis=1)

        u = self.prepare.equipment_first_order_edges
        u["var"] = u.apply(lambda r: model.NewBoolVar(f"u_{r.equipment_id}_{r.leg_id}"), axis=1)

        y = self.prepare.equipment_order_order_edges
        y["var"] = y.apply(
            lambda r: model.NewBoolVar(f"y_{r.equipment_id}_{r.leg_id_from}_{r.leg_id_to}"), axis=1
        )

        self._start = self.data.orders.set_index("leg_id").apply(
            lambda r: model.NewIntVar(
                self._scale(r.time_window_start), self._scale(r.time_window_end - r.duration_hours),
                f"start_{r.name}",
            ),
            axis=1,
        )

    def _add_one_equip_per_order_constraints(self) -> None:
        """
            (1) Не более одной единицы техники на ногу заявки.
        """

        model = self._model
        x = self.prepare.equipment_order_edges
        x.groupby("leg_id")["var"].apply(lambda g: model.Add(sum(g.tolist()) <= 1))

    def _add_one_first_job_constraints(self) -> None:
        """
            (2) Не более одной "первой работы" на единицу техники.
        """

        model = self._model
        u = self.prepare.equipment_first_order_edges
        u.groupby("equipment_id")["var"].apply(lambda g: model.Add(sum(g.tolist()) <= 1))

    def _compute_flow_incidence(self) -> None:
        """
            Списки входящих/исходящих u/y переменных на каждый
            (equipment_id, leg_id) — общая подготовка для (3) и (4).
        """

        x = self.prepare.equipment_order_edges
        u = self.prepare.equipment_first_order_edges
        y = self.prepare.equipment_order_order_edges

        inflow = pd.concat([
            u[["equipment_id", "leg_id", "var"]],
            y.rename(columns={"leg_id_to": "leg_id"})[["equipment_id", "leg_id", "var"]],
        ]).groupby(["equipment_id", "leg_id"])["var"].apply(list)
        outflow = y.rename(columns={"leg_id_from": "leg_id"})[["equipment_id", "leg_id", "var"]] \
            .groupby(["equipment_id", "leg_id"])["var"].apply(list)

        x["in_vars"] = x.apply(lambda r: inflow.get((r.equipment_id, r.leg_id), []), axis=1)
        x["out_vars"] = x.apply(lambda r: outflow.get((r.equipment_id, r.leg_id), []), axis=1)

    def _add_inflow_constraints(self) -> None:
        """
            (3) Баланс потока (вход = назначение).
        """

        model = self._model
        x = self.prepare.equipment_order_edges
        x.apply(
            lambda r: model.Add(sum(r.in_vars) == r["var"]) if r.in_vars else model.Add(r["var"] == 0),
            axis=1,
        )

    def _add_outflow_constraints(self) -> None:
        """
            (4) Баланс потока (выход <= назначение).
        """

        model = self._model
        x = self.prepare.equipment_order_edges
        x[x["out_vars"].map(len) > 0].apply(
            lambda r: model.Add(sum(r.out_vars) <= r["var"]),
            axis=1,
        )

    def _add_availability_constraints(self) -> None:
        """
            (5) Работы должны завершиться до конца доступности техники.
            OnlyEnforceIf вместо Big-M — ограничение активно, только если
            единица e реально назначена на ногу (x[e,leg] = 1).
        """

        model = self._model
        start = self._start
        x = self.prepare.equipment_order_edges
        x.apply(
            lambda r: model.Add(
                start[r.leg_id] + self._scale(r.duration_hours) <= self._scale(r.available_until)
            ).OnlyEnforceIf(r["var"]),
            axis=1,
        )

    def _add_chain_timing_constraints(self) -> None:
        """
            (6) Согласованность времени в цепочке (OnlyEnforceIf вместо Big-M).
        """

        model = self._model
        start = self._start
        y = self.prepare.equipment_order_order_edges
        y.apply(
            lambda r: model.Add(
                start[r.leg_id_to] >= start[r.leg_id_from] + self._scale(r.duration_from + r.travel_h)
            ).OnlyEnforceIf(r["var"]),
            axis=1,
        )

    def _add_first_job_timing_constraints(self) -> None:
        """
            (7) Инициализация времени начала выполнения первой заявки (OnlyEnforceIf вместо Big-M).
        """

        model = self._model
        start = self._start
        u = self.prepare.equipment_first_order_edges
        u.apply(
            lambda r: model.Add(
                start[r.leg_id] >= self._scale(r.available_from + r.travel_h)
            ).OnlyEnforceIf(r["var"]),
            axis=1,
        )

    def _add_combo_constraints(self) -> None:
        """
            (8) Комбинированные заказы: несколько строк (ног) с одинаковым
            order_id — это один и тот же физический заказ, которому
            одновременно нужно несколько типов техники. Ноги одного заказа
            либо выполняются ВСЕ разом, либо ни одна, и стартуют в один и тот
            же момент — иначе техника работает не вместе, а сама по себе.
        """

        model = self._model
        start = self._start
        x = self.prepare.equipment_order_edges

        # сумма x по каждой ноге — сколько единиц техники её выполняет (0 или 1)
        leg_sum = x.groupby("leg_id")["var"].apply(lambda g: sum(g.tolist()))

        legs = x[["order_id", "leg_id"]].drop_duplicates()
        combo_pairs = legs.merge(legs, on="order_id", suffixes=("_a", "_b"))
        combo_pairs = combo_pairs[combo_pairs["leg_id_a"] < combo_pairs["leg_id_b"]]

        combo_pairs.apply(lambda r: model.Add(leg_sum[r.leg_id_a] == leg_sum[r.leg_id_b]), axis=1)
        combo_pairs.apply(lambda r: model.Add(start[r.leg_id_a] == start[r.leg_id_b]), axis=1)

    def _add_no_overlap_constraints(self) -> None:
        """
            Избыточное (по транзитивности цепочки y — как в constraint (6)),
            но полезное ограничение: одна единица техники не может быть на
            двух заявках одновременно. У чистого MIP (Model) такого нет — там
            это следует из баланса потока (3)/(4); здесь это отдельный
            нативный для CP-SAT примитив, который даёт solveru более сильную
            propagation, чем эквивалентная линейная запись.
        """

        model = self._model
        x = self.prepare.equipment_order_edges

        x["interval"] = x.apply(
            lambda r: model.NewOptionalFixedSizeIntervalVar(
                self._start[r.leg_id], self._scale(r.duration_hours), r["var"], f"iv_{r.equipment_id}_{r.leg_id}"
            ),
            axis=1,
        )
        x.groupby("equipment_id")["interval"].apply(lambda g: model.AddNoOverlap(g.tolist()))

    def _add_objective(self) -> None:
        """
            Максимизировать сумму весов выполненных заказов минус суммарные
            затраты на перегон (к первой работе + между заказами в цепочке).
            У всех ног заказа, кроме первой, вес выполнения — 0: constraint
            (8) и так гарантирует, что они выполняются вместе с первой, иначе
            один и тот же реальный заказ засчитался бы в ЦФ несколько раз.
            Коэффициенты округляются до целых — CP-SAT не принимает дробные
            веса в целевой функции.
        """

        model = self._model
        orders = self.data.orders.copy()
        x = self.prepare.equipment_order_edges
        u = self.prepare.equipment_first_order_edges
        y = self.prepare.equipment_order_order_edges

        orders["is_primary_leg"] = ~orders.duplicated("order_id", keep="first")
        orders["order_weight"] = orders["is_mandatory"].map(
            {True: config.MANDATORY_ORDER_PENALTY_RUB, False: config.UNASSIGNED_ORDER_PENALTY_RUB}
        )
        orders.loc[~orders["is_primary_leg"], "order_weight"] = 0.0
        weight_map = orders.set_index("leg_id")["order_weight"]
        weight = x["leg_id"].map(weight_map).round().astype(int)

        fulfillment_term = sum((x["var"] * weight).tolist())
        travel_term = sum((u["var"] * u["cost"].round().astype(int)).tolist()) \
            + sum((y["var"] * y["cost"].round().astype(int)).tolist())

        model.Maximize(fulfillment_term - travel_term)

    def _extract_assignment(self) -> pd.DataFrame:
        """
            Принятые назначения x[e,leg] = 1 вместе с найденным временем старта.
            order_id остаётся для читаемости — у комбинированных заказов
            несколько строк с одним order_id и разным equipment_id/leg_id.
        """

        solver = self._solver
        x = self.prepare.equipment_order_edges
        assigned = x[x["var"].apply(lambda v: solver.Value(v) > 0.5)].copy()
        assigned["start_time_h"] = assigned["leg_id"].apply(
            lambda leg: round(solver.Value(self._start[leg]) / TIME_SCALE, 2)
        )

        return assigned[["equipment_id", "order_id", "leg_id", "start_time_h"]] \
            .sort_values(["equipment_id", "start_time_h"]).reset_index(drop=True)
