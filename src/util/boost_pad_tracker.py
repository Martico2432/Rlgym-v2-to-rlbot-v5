from dataclasses import dataclass

from rlbot.flat import FieldInfo, GamePacket

from util.vec import Vec3


@dataclass
class BoostPad:
    location: Vec3
    is_full_boost: bool
    is_active: bool  # Active means it's available to be picked up
    timer: float  # Counts the number of seconds that the pad has been *inactive*


class BoostPadTracker:
    """
    This class merges together the boost pad location info with the is_active info so you can access it
    in one convenient list. For it to function correctly, you need to call initialize_boosts once when the
    game has started, and then update_boost_status every frame so that it knows which pads are active.
    """

    def __init__(self):
        self.boost_pads: list[BoostPad] = []
        self._full_boosts_only: list[BoostPad] = []

    def initialize_boosts(self, game_info: FieldInfo):
        self.boost_pads: list[BoostPad] = [
            BoostPad(Vec3(rb.location), rb.is_full_boost, False, 0)
            for rb in game_info.boost_pads
        ]

        # Cache the list of full boosts since they're commonly requested.
        # They reference the same objects in the boost_pads list.
        self._full_boosts_only: list[BoostPad] = [
            bp for bp in self.boost_pads if bp.is_full_boost
        ]

    def update_boost_status(self, packet: GamePacket):
        for i, packet_pad in enumerate(packet.boost_pads):
            our_pad = self.boost_pads[i]
            our_pad.is_active = packet_pad.is_active
            our_pad.timer = packet_pad.timer

    def get_full_boosts(self) -> list[BoostPad]:
        return self._full_boosts_only
