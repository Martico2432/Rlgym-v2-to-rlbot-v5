from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ExtraPlayerInfo:
    on_ground: bool
    handbrake: float
    ball_touches: int
    car_contact_id: int
    car_contact_cooldown_timer: float
    is_autoflipping: bool
    autoflip_timer: float
    autoflip_direction: float  # 1 or -1, determines roll direction


@dataclass
class ExtraBallInfo:
    # Net that the heatseeker ball is targeting (0 for none, 1 for orange, -1 for blue)
    heatseeker_target_dir: int
    heatseeker_target_speed: float
    heatseeker_time_since_hit: float


@dataclass
class ExtraPacketInfo:
    players: List[ExtraPlayerInfo]
    ball: ExtraBallInfo
