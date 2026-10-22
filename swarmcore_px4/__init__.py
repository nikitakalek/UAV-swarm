"""Адаптер PX4 (M2-11, M2-12): тот же ``MotionSource``, но поверх MAVSDK.

Логика роя (``swarmsim``) не знает, летит ли агент в кинематической модели или
в PX4 SITL / на реальном борту. Требует ``pip install "mavsdk<4"``.
"""

from swarmcore_px4.mavsdk_motion import MavsdkMotion, sitl_address

__all__ = ["MavsdkMotion", "sitl_address"]
