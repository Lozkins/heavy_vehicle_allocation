"""MIP-модель назначения техники на заявки (HiGHS через highspy)."""

import time as time_module

import highspy
import pandas as pd

import config
from prepare import Prepare

_STATUS_NAMES = {
    highspy.HighsModelStatus.kOptimal: "Optimal",
    highspy.HighsModelStatus.kInfeasible: "Infeasible",
    highspy.HighsModelStatus.kTimeLimit: "TimeLimit",
}


class Model:
    """Строит и решает MIP-модель назначения по таблицам из Prepare."""

    def __init__(self, prepare: Prepare, time_limit_sec: float = 300.0, mip_rel_gap: float = 0.005):
        self.prepare = prepare.clone()
        self.data = self.prepare.data
        self.time_limit_sec = time_limit_sec
        self.mip_rel_gap = mip_rel_gap

        self._big_m = self._compute_big_m()
        self._h = highspy.Highs()
        self._start = None
        self._compiled = False

    def compile(self) -> "Model":
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

        self._h.setOptionValue("output_flag", True)
        self._h.setOptionValue("time_limit", float(self.time_limit_sec))
        self._h.setOptionValue("mip_rel_gap", float(self.mip_rel_gap))

        t0 = time_module.time()
        self._h.run()
        self.solve_time_sec = time_module.time() - t0

        model_status = self._h.getModelStatus()
        self.status = _STATUS_NAMES.get(model_status, str(model_status))
        self.objective_value = self._h.getObjectiveValue()
        self.assignment = self._extract_assignment()

    def _compute_big_m(self) -> float:
        """
            Достаточно большая константа для "выключающих" ограничений —
            больше суммы горизонта планирования и максимального времени в
            пути; вычисляется из данных, а не задаётся вручную.
        """

        orders = self.data.orders
        horizon = orders["time_window_end"].max() - orders["time_window_start"].min()

        return float(horizon + self.data.duration.values.max())

    def _add_variables(self) -> None:
        """
            Создаёт x[e,leg], u[e,leg], y[e,leg1,leg2] и s_leg. Переменные
            дописываются колонкой var прямо в таблицы Prepare — та же таблица
            дальше используется и в ограничениях, и в целевой функции.
        """

        h = self._h

        x = self.prepare.equipment_order_edges
        x["var"] = x.apply(
            lambda r: h.addVariable(0, 1, type=highspy.HighsVarType.kInteger, name=f"x_{r.equipment_id}_{r.leg_id}"),
            axis=1,
        )

        u = self.prepare.equipment_first_order_edges
        u["var"] = u.apply(
            lambda r: h.addVariable(0, 1, type=highspy.HighsVarType.kInteger, name=f"u_{r.equipment_id}_{r.leg_id}"),
            axis=1,
        )

        y = self.prepare.equipment_order_order_edges
        y["var"] = y.apply(
            lambda r: h.addVariable(
                0, 1, type=highspy.HighsVarType.kInteger,
                name=f"y_{r.equipment_id}_{r.leg_id_from}_{r.leg_id_to}",
            ),
            axis=1,
        )

        self._start = self.data.orders.set_index("leg_id").apply(
            lambda r: h.addVariable(r.time_window_start, r.time_window_end - r.duration_hours, name=f"start_{r.name}"),
            axis=1,
        )

    def _add_one_equip_per_order_constraints(self) -> None:
        """
            (1) Не более одной единицы техники на ногу заявки.
        """

        h = self._h
        x = self.prepare.equipment_order_edges
        x.groupby("leg_id")["var"].apply(
            lambda g: h.addConstr(h.qsum(g.tolist()) <= 1, name=f"one_equip_per_order_{g.name}")
        )

    def _add_one_first_job_constraints(self) -> None:
        """
            (2) Не более одной "первой работы" на единицу техники.
        """

        h = self._h
        u = self.prepare.equipment_first_order_edges
        u.groupby("equipment_id")["var"].apply(
            lambda g: h.addConstr(h.qsum(g.tolist()) <= 1, name=f"one_first_job_{g.name}")
        )

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

        h = self._h
        x = self.prepare.equipment_order_edges
        x.apply(
            lambda r: h.addConstr(
                h.qsum(r.in_vars) == r["var"] if r.in_vars else r["var"] == 0,
                name=f"inflow_{r.equipment_id}_{r.leg_id}",
            ),
            axis=1,
        )

    def _add_outflow_constraints(self) -> None:
        """
            (4) Баланс потока (выход <= назначение).
        """

        h = self._h
        x = self.prepare.equipment_order_edges
        x[x["out_vars"].map(len) > 0].apply(
            lambda r: h.addConstr(h.qsum(r.out_vars) <= r["var"], name=f"outflow_{r.equipment_id}_{r.leg_id}"),
            axis=1,
        )

    def _add_availability_constraints(self) -> None:
        """
            (5) Работы должны завершиться до конца доступности техники.
        """

        h = self._h
        m = self._big_m
        start = self._start
        x = self.prepare.equipment_order_edges
        x.apply(
            lambda r: h.addConstr(
                start[r.leg_id] + r.duration_hours <= r.available_until + m * (1 - r["var"]),
                name=f"avail_until_{r.equipment_id}_{r.leg_id}",
            ),
            axis=1,
        )

    def _add_chain_timing_constraints(self) -> None:
        """
            (6) Согласованность времени в цепочке.
        """

        h = self._h
        m = self._big_m
        start = self._start
        y = self.prepare.equipment_order_order_edges
        y.apply(
            lambda r: h.addConstr(
                start[r.leg_id_to] >= start[r.leg_id_from] + r.duration_from + r.travel_h - m * (1 - r["var"]),
                name=f"seq_time_{r.equipment_id}_{r.leg_id_from}_{r.leg_id_to}",
            ),
            axis=1,
        )

    def _add_first_job_timing_constraints(self) -> None:
        """
            (7) Инициализация времени начала выполнения первой заявки.
        """

        h = self._h
        m = self._big_m
        start = self._start
        u = self.prepare.equipment_first_order_edges
        u.apply(
            lambda r: h.addConstr(
                start[r.leg_id] >= r.available_from + r.travel_h - m * (1 - r["var"]),
                name=f"first_time_{r.equipment_id}_{r.leg_id}",
            ),
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

        h = self._h
        start = self._start
        x = self.prepare.equipment_order_edges

        # сумма x по каждой ноге — сколько единиц техники её выполняет (0 или 1)
        leg_sum = x.groupby("leg_id")["var"].apply(lambda g: h.qsum(g.tolist()))

        legs = x[["order_id", "leg_id"]].drop_duplicates()
        combo_pairs = legs.merge(legs, on="order_id", suffixes=("_a", "_b"))
        combo_pairs = combo_pairs[combo_pairs["leg_id_a"] < combo_pairs["leg_id_b"]]

        combo_pairs.apply(
            lambda r: h.addConstr(
                leg_sum[r.leg_id_a] == leg_sum[r.leg_id_b], name=f"combo_link_{r.leg_id_a}_{r.leg_id_b}"
            ),
            axis=1,
        )
        combo_pairs.apply(
            lambda r: h.addConstr(
                start[r.leg_id_a] == start[r.leg_id_b], name=f"combo_time_{r.leg_id_a}_{r.leg_id_b}"
            ),
            axis=1,
        )

    def _add_objective(self) -> None:
        """
            Максимизировать сумму весов выполненных заказов минус суммарные
            затраты на перегон (к первой работе + между заказами в цепочке).
            У всех ног заказа, кроме первой, вес выполнения — 0: constraint
            (8) и так гарантирует, что они выполняются вместе с первой, иначе
            один и тот же реальный заказ засчитался бы в ЦФ несколько раз.
        """

        h = self._h
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
        weight = x["leg_id"].map(weight_map)

        fulfillment_term = h.qsum((x["var"] * weight).tolist())
        travel_term = h.qsum((u["var"] * u["cost"]).tolist()) + h.qsum((y["var"] * y["cost"]).tolist())

        h.setObjective(fulfillment_term - travel_term)
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    def _extract_assignment(self) -> pd.DataFrame:
        """
            Принятые назначения x[e,leg] = 1 вместе с найденным временем старта.
            order_id остаётся для читаемости — у комбинированных заказов
            несколько строк с одним order_id и разным equipment_id/leg_id.
        """

        h = self._h
        x = self.prepare.equipment_order_edges
        assigned = x[x["var"].apply(lambda v: h.val(v) > 0.5)].copy()
        assigned["start_time_h"] = assigned["leg_id"].apply(lambda leg: round(h.val(self._start[leg]), 2))

        return assigned[["equipment_id", "order_id", "leg_id", "start_time_h"]] \
            .sort_values(["equipment_id", "start_time_h"]).reset_index(drop=True)
    
