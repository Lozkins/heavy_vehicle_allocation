"""Подготовка входных данных для модели: затраты на перегон и допустимые рёбра."""

import pandas as pd

import config
from data import Data


class Prepare:
    """Считает производные таблицы поверх Data — то, что дальше пойдёт в модель."""

    def __init__(self, data: Data):
        self.data = data
        self.mobilization_cost = self._compute_mobilization_cost()
        self.equipment_order_edges = self._compute_equipment_order_edges()
        self.equipment_first_order_edges = self._compute_equipment_first_order_edges()
        self.equipment_order_order_edges = self._compute_equipment_order_order_edges()

    def clone(self) -> "Prepare":
        """
            Независимая копия (то же self.data, свои DataFrame'ы). Нужна,
            когда один и тот же Prepare идёт в несколько Model — Model
            дописывает колонку var прямо в таблицы рёбер, и без клона второй
            Model унаследовал бы переменные первого.
        """

        return Prepare(self.data)

    def _compute_mobilization_cost(self) -> pd.DataFrame:
        """
            Матрица затрат на перегон, ₽; та же форма, что и матрица расстояний.
            Ноль, если перегон не требуется (расстояние ~0 — техника уже на месте).
        """

        distance = self.data.distance
        fixed_fee = (config.MOBILIZATION_FIXED_FEE_MIN + config.MOBILIZATION_FIXED_FEE_MAX) / 2
        cost = distance * config.MOBILIZATION_RATE_PER_KM + fixed_fee

        return cost.where(distance > 1e-6, 0.0)

    def _compute_equipment_order_edges(self) -> pd.DataFrame:
        """
            Допустимые рёбра техника-заявка (по leg_id — одной "ноге" заявки,
            см. Data._load_orders):
             - совпадает тип,
             - класс техники не ниже требуемого.

            Временные окна здесь не проверяются — это домен переменной назначения x[e,o],
            а не готовое расписание. С собой несёт order_id (для связки ног
            комбинированных заказов друг с другом), available_until и
            duration_hours — это всё, что понадобится модели дальше по (e, leg),
            без повторных merge.
        """

        fleet = self.data.fleet.copy()
        orders = self.data.orders.copy()

        capacity_rank = {c: i for i, c in enumerate(config.CAPACITY_CLASSES)}
        fleet["capacity_rank"] = fleet["capacity_class"].map(capacity_rank)
        orders["min_capacity_rank"] = orders["min_capacity_class"].map(capacity_rank)

        pairs = fleet[["equipment_id", "type", "capacity_rank", "available_until"]].merge(
            orders[["leg_id", "order_id", "required_type", "min_capacity_rank", "duration_hours"]], how="cross",
        )
        mask = (pairs["type"] == pairs["required_type"]) & (pairs["capacity_rank"] >= pairs["min_capacity_rank"])

        return pairs.loc[mask, ["equipment_id", "leg_id", "order_id", "available_until", "duration_hours"]] \
            .reset_index(drop=True)

    def _compute_equipment_first_order_edges(self) -> pd.DataFrame:
        """
            Допустимые рёбра "первая работа": единица e совместима с ногой leg
            (equipment_order_edges) И успевает доехать из депо до начала окна
            leg и стартовать не позже конца окна (единица ещё не работала —
            едет из депо, а не с предыдущей заявки). С собой несёт
            available_from, travel_h (депо -> заявка) и cost (перегон депо -> заявка).
        """

        fleet = self.data.fleet[["equipment_id", "depot_location", "available_from"]]
        orders = self.data.orders[["leg_id", "location", "time_window_end", "duration_hours"]]

        edges = self.equipment_order_edges[["equipment_id", "leg_id"]] \
            .merge(fleet, on="equipment_id").merge(orders, on="leg_id")

        travel = self.data.duration.stack().rename("travel_h").reset_index()
        travel.columns = ["depot_location", "location", "travel_h"]
        edges = edges.merge(travel, on=["depot_location", "location"])

        cost = self.mobilization_cost.stack().rename("cost").reset_index()
        cost.columns = ["depot_location", "location", "cost"]
        edges = edges.merge(cost, on=["depot_location", "location"])

        earliest_arrival = edges["available_from"] + edges["travel_h"]
        latest_start = edges["time_window_end"] - edges["duration_hours"]
        feasible = earliest_arrival <= latest_start

        return edges.loc[feasible, ["equipment_id", "leg_id", "available_from", "travel_h", "cost"]] \
            .reset_index(drop=True)

    def _compute_leg_pair_feasibility(self) -> pd.DataFrame:
        """
            Пары ног заявок (leg1, leg2), для которых в принципе успевает
            ЛЮБАЯ единица техники: leg1 стартует в САМОЕ РАННЕЕ возможное
            время своего окна, и даже тогда единица успевает доехать до leg2
            и начать её не позже конца её окна. Необходимое (не достаточное)
            условие — не зависит от конкретной единицы техники, только от
            окон/длительностей заявок и времени в пути между ними. Пары ног
            ОДНОГО И ТОГО ЖЕ заказа (order_id) исключены — это не
            последовательная работа, а одновременная (см. constraint (8) в
            Model). С собой несёт duration_from, travel_h и cost (перегон
            между локациями заявок).
        """

        orders = self.data.orders
        duration = self.data.duration

        left = orders[["leg_id", "order_id", "location", "time_window_start", "duration_hours"]].rename(
            columns={"leg_id": "leg_id_from", "order_id": "order_id_from", "location": "location_from",
                     "time_window_start": "start_from", "duration_hours": "duration_from"}
        )
        right = orders[["leg_id", "order_id", "location", "time_window_end", "duration_hours"]].rename(
            columns={"leg_id": "leg_id_to", "order_id": "order_id_to", "location": "location_to",
                     "time_window_end": "end_to", "duration_hours": "duration_to"}
        )
        pairs = left.merge(right, how="cross")
        pairs = pairs[pairs["order_id_from"] != pairs["order_id_to"]]

        travel = duration.stack().rename("travel_h").reset_index()
        travel.columns = ["location_from", "location_to", "travel_h"]
        pairs = pairs.merge(travel, on=["location_from", "location_to"])

        cost = self.mobilization_cost.stack().rename("cost").reset_index()
        cost.columns = ["location_from", "location_to", "cost"]
        pairs = pairs.merge(cost, on=["location_from", "location_to"])

        earliest_finish = pairs["start_from"] + pairs["duration_from"]
        latest_start_to = pairs["end_to"] - pairs["duration_to"]
        feasible = earliest_finish + pairs["travel_h"] <= latest_start_to

        return pairs.loc[feasible, ["leg_id_from", "leg_id_to", "duration_from", "travel_h", "cost"]] \
            .reset_index(drop=True)

    def _compute_equipment_order_order_edges(self) -> pd.DataFrame:
        """
            Допустимые рёбра техника нога-нога: пара ног физически успеваема
            (см. _compute_leg_pair_feasibility) И обе ноги пары совместимы с
            одной и той же единицей техники (equipment_order_edges).
        """

        leg_pairs = self._compute_leg_pair_feasibility()
        eq_leg = self.equipment_order_edges[["equipment_id", "leg_id"]]

        merged = leg_pairs.merge(
            eq_leg.rename(columns={"leg_id": "leg_id_from"}), on="leg_id_from"
        ).merge(
            eq_leg.rename(columns={"leg_id": "leg_id_to"}), on=["equipment_id", "leg_id_to"]
        )

        return merged[["equipment_id", "leg_id_from", "leg_id_to", "duration_from", "travel_h", "cost"]] \
            .reset_index(drop=True)
