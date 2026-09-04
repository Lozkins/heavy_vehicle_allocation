"""Жадный (наивный) алгоритм назначения техники на заявки — baseline для Model."""

import pandas as pd

from prepare import Prepare


class Greedy:
    """
        Обрабатывает заказы по одному (сначала обязательные, потом по
        дедлайну) и ставит на каждый ближайшую по затратам на перегон
        единицу техники, которая физически успевает. Комбинированные заказы
        (несколько строк с одним order_id) назначаются целиком одной парой
        единиц техники сразу или не назначаются вовсе.
    """

    def __init__(self, prepare: Prepare):
        self.prepare = prepare
        self.data = prepare.data

    def run(self) -> "Greedy":
        """
            Запускает жадное назначение техники на заявки. После вызова
            доступны self.assignment и self.unassigned.
        """

        self._state = self._init_state()
        self._order_queue = self._build_order_queue()
        self.assignment = self._assign_all()

        return self

    def _init_state(self) -> pd.DataFrame:
        """
            Текущее положение и момент высвобождения каждой единицы техники, индекс — equipment_id.
        """

        fleet = self.data.fleet.set_index("equipment_id")

        return fleet[["type", "capacity_class", "depot_location", "available_from", "available_until"]].rename(
            columns={"depot_location": "location", "available_from": "free_time"}
        )

    def _build_order_queue(self) -> pd.DataFrame:
        """
            Одна строка на order_id (не на ногу), отсортированная сначала по
            обязательности, потом по дедлайну. Колонка legs — список
            (leg_id, required_type, min_capacity_class): один элемент для
            обычного заказа, два — для комбинированного.
        """

        orders = self.data.orders
        shared = orders[
            ["order_id", "location", "time_window_start", "time_window_end", "duration_hours", "is_mandatory"]
        ].drop_duplicates("order_id")

        legs = orders.groupby("order_id").apply(
            lambda g: list(zip(g["leg_id"], g["required_type"], g["min_capacity_class"])), include_groups=False
        ).rename("legs").reset_index()

        queue = shared.merge(legs, on="order_id")

        return queue.sort_values(["is_mandatory", "time_window_end"], ascending=[False, True]).reset_index(drop=True)

    def _feasible_candidates(self, leg_id: str, location: str, window_start: float, window_end: float,
                              duration_hours: float) -> pd.DataFrame:
        """
            Единицы техники, совместимые с ногой leg_id (Prepare.equipment_order_edges
            уже отфильтровал по типу/классу) И физически успевающие приехать
            из текущего положения и начать работу в пределах окна.
        """

        eligible = self.prepare.equipment_order_edges.loc[
            self.prepare.equipment_order_edges["leg_id"] == leg_id, "equipment_id"
        ]
        candidates = self._state.loc[eligible].reset_index()
        if candidates.empty:
            return candidates

        candidates["travel_h"] = candidates["location"].apply(lambda loc: float(self.data.duration.loc[loc, location]))
        candidates["cost"] = candidates["location"].apply(
            lambda loc: float(self.prepare.mobilization_cost.loc[loc, location])
        )
        candidates["earliest_arrival"] = candidates["free_time"] + candidates["travel_h"]
        candidates["actual_start"] = candidates["earliest_arrival"].clip(lower=window_start)
        candidates["finish"] = candidates["actual_start"] + duration_hours

        feasible = (candidates["finish"] <= window_end) & (candidates["finish"] <= candidates["available_until"])

        return candidates[feasible].reset_index(drop=True)

    def _commit(self, equipment_id: str, location: str, start_time: float, duration_hours: float) -> None:
        """
            Переводит единицу техники в новое положение и момент высвобождения после выполнения работы.
        """

        self._state.loc[equipment_id, "location"] = location
        self._state.loc[equipment_id, "free_time"] = start_time + duration_hours

    def _assign_single(self, order_row) -> list:
        """
            Обычный заказ (одна нога) — берём совместимую единицу с минимальными затратами на перегон.
        """

        leg_id, _required_type, _min_capacity_class = order_row.legs[0]
        candidates = self._feasible_candidates(
            leg_id, order_row.location, order_row.time_window_start, order_row.time_window_end,
            order_row.duration_hours,
        )
        if candidates.empty:
            return []

        best = candidates.loc[candidates["cost"].idxmin()]
        self._commit(best.equipment_id, order_row.location, best.actual_start, order_row.duration_hours)

        return [{
            "equipment_id": best.equipment_id, "order_id": order_row.order_id, "leg_id": leg_id,
            "start_time_h": round(best.actual_start, 2), "travel_h": round(best.travel_h, 2),
            "mobilization_cost_rub": round(best.cost, 2),
        }]

    def _assign_combo(self, order_row) -> list:
        """
            Комбинированный заказ (две ноги) — перебираем все пары
            (кандидат ноги 1) x (кандидат ноги 2), обе должны стартовать в
            ОДИН момент времени (максимум из их earliest_arrival и начала
            окна), выбираем пару с минимальной суммой затрат на перегон.
        """

        (leg_id_1, _t1, _c1), (leg_id_2, _t2, _c2) = order_row.legs
        c1 = self._feasible_candidates(
            leg_id_1, order_row.location, order_row.time_window_start, order_row.time_window_end,
            order_row.duration_hours,
        )
        c2 = self._feasible_candidates(
            leg_id_2, order_row.location, order_row.time_window_start, order_row.time_window_end,
            order_row.duration_hours,
        )
        if c1.empty or c2.empty:
            return []

        pairs = c1.merge(c2, how="cross", suffixes=("_1", "_2"))
        pairs["actual_start"] = pairs[["earliest_arrival_1", "earliest_arrival_2"]] \
            .max(axis=1).clip(lower=order_row.time_window_start)
        pairs["finish"] = pairs["actual_start"] + order_row.duration_hours

        feasible = pairs[
            (pairs["finish"] <= order_row.time_window_end)
            & (pairs["finish"] <= pairs["available_until_1"])
            & (pairs["finish"] <= pairs["available_until_2"])
        ]
        if feasible.empty:
            return []

        best = feasible.loc[(feasible["cost_1"] + feasible["cost_2"]).idxmin()]
        self._commit(best.equipment_id_1, order_row.location, best.actual_start, order_row.duration_hours)
        self._commit(best.equipment_id_2, order_row.location, best.actual_start, order_row.duration_hours)

        return [
            {"equipment_id": best.equipment_id_1, "order_id": order_row.order_id, "leg_id": leg_id_1,
             "start_time_h": round(best.actual_start, 2), "travel_h": round(best.travel_h_1, 2),
             "mobilization_cost_rub": round(best.cost_1, 2)},
            {"equipment_id": best.equipment_id_2, "order_id": order_row.order_id, "leg_id": leg_id_2,
             "start_time_h": round(best.actual_start, 2), "travel_h": round(best.travel_h_2, 2),
             "mobilization_cost_rub": round(best.cost_2, 2)},
        ]

    def _assign_all(self) -> pd.DataFrame:
        """
            Проходит очередь заказов, назначает каждый и копит неудачные попытки в self.unassigned.
        """

        rows = []
        unassigned = []
        for order_row in self._order_queue.itertuples(index=False):
            result = self._assign_single(order_row) if len(order_row.legs) == 1 else self._assign_combo(order_row)
            if result:
                rows += result
            else:
                unassigned.append({"order_id": order_row.order_id, "is_mandatory": order_row.is_mandatory})

        self.unassigned = pd.DataFrame(unassigned)

        return pd.DataFrame(rows)
