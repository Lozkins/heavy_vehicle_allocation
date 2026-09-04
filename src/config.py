"""Конфигурация проекта."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Классы техники по грузоподъёмности/мощности — иерархия: единица класса X
# подходит под заказ, требующий класс <= X (порядок слева направо).
CAPACITY_CLASSES = ["light", "medium", "heavy"]

# Тариф перегона (мобилизации) техники между объектами: ₽/км по факту
# пройденного пути + фиксированная плата за погрузку/разгрузку на трал.
MOBILIZATION_RATE_PER_KM = 80.0
MOBILIZATION_FIXED_FEE_MIN = 5000.0
MOBILIZATION_FIXED_FEE_MAX = 8000.0

# Вес выполнения заказа в целевой функции модели: обычный (рыночный) заказ.
UNASSIGNED_ORDER_PENALTY_RUB = 1_000_000.0

# Вес выполнения сервисного (обязательного) заказа — на порядок больше, чтобы
# решатель не пожертвовал им ради нескольких обычных заказов разом.
MANDATORY_ORDER_PENALTY_RUB = 20.0 * UNASSIGNED_ORDER_PENALTY_RUB
