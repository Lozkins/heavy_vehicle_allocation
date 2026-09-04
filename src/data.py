"""Считывание входных данных в pandas."""

from pathlib import Path

import pandas as pd

from config import DATA_DIR


class Data:
    """Загружает входные данные эксперимента и хранит их как DataFrame'ы."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.fleet = self._load_fleet()
        self.orders = self._load_orders()
        self.locations = self._load_locations()
        self.distance = self._load_distance_matrix()
        self.duration = self._load_duration_matrix()

    def _load_fleet(self) -> pd.DataFrame:
        """
            Парк техники: equipment_id, type, capacity_class, depot_location, available_from/until, hourly_rate_rub.
        """

        return pd.read_csv(self.data_dir / "fleet.csv")

    def _load_orders(self) -> pd.DataFrame:
        """
            Заявки: order_id, required_type, min_capacity_class, location, time_window_start/end,
            duration_hours, urgency, is_mandatory, is_long_term.

            order_id не уникален: две строки с одинаковым order_id — это один
            комбинированный заказ, где одновременно нужны две единицы техники
            разных типов (required_type у строк разный). leg_id — отдельный,
            уже гарантированно уникальный идентификатор СТРОКИ (одной "ноги"
            заказа) — на нём дальше строятся переменные и рёбра модели,
            order_id остаётся только для связки ног друг с другом.
        """

        orders = pd.read_csv(self.data_dir / "orders.csv")
        orders["leg_id"] = orders["order_id"] + "#" + (orders.groupby("order_id").cumcount() + 1).astype(str)

        return orders

    def _load_locations(self) -> pd.DataFrame:
        """
            Точки (депо + объекты): location_id, kind, address, city, lat, lon, geocode_source.
        """

        return pd.read_csv(self.data_dir / "locations.csv")

    def _load_distance_matrix(self) -> pd.DataFrame:
        """
            Квадратная матрица расстояний по дороге, км; индекс и колонки — location_id.
        """

        return pd.read_csv(self.data_dir / "distance_km.csv", index_col=0)

    def _load_duration_matrix(self) -> pd.DataFrame:
        """
            Квадратная матрица времени в пути, часы; индекс и колонки — location_id.
        """

        return pd.read_csv(self.data_dir / "duration_h.csv", index_col=0)
