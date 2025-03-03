from collections import deque
from typing import Dict

import RocketSim as rsim
from rlbot.flat import (
    BallTypeMutator,
    BoostMutator,
    BoostStrengthMutator,
    DemolishMutator,
    FieldInfo,
    GameEventMutator,
    GameMode,
    GamePacket,
    GameSpeedMutator,
    GravityMutator,
    MatchConfiguration,
    MultiBallMutator,
    PlayerInfo,
    RespawnTimeMutator,
    RumbleMutator
)

from .car import Car
from .extra_info import ExtraBallInfo, ExtraPacketInfo, ExtraPlayerInfo
from .math import euler_to_rotation
from .utils import rotator_to_numpy, vector_to_numpy


class SimExtraInfo:
    def __init__(
        self, field_info: FieldInfo, match_settings=MatchConfiguration(), tick_skip=8
    ):
        match match_settings.game_mode:
            case GameMode.Soccer:
                mode = rsim.GameMode.SOCCAR
            case GameMode.Hoops:
                mode = rsim.GameMode.HOOPS
            case GameMode.Heatseeker:
                mode = rsim.GameMode.HEATSEEKER
            case GameMode.Hockey:
                mode = rsim.GameMode.SNOWDAY
            case _:
                raise NotImplementedError(match_settings.game_mode)
        # TODO: ensure the boost pads are right

        # Ensure there are no mutators configured that we can't support
        mutators = match_settings.mutators
        if mutators is not None:
            mutator_config = {}
            assert (
                mutators.multi_ball == MultiBallMutator.One
            ), "Can only use one ball with sim"

            assert (
                mutators.game_speed == GameSpeedMutator.Default
            ), "Can only use default game speed with sim"

            match mutators.ball_type:
                case BallTypeMutator.Default:
                    assert (
                        match_settings.game_mode == GameMode.Soccer
                    ), "Cannot use non-soccer ball in soccer with sim"
                case BallTypeMutator.Puck:
                    assert (
                        match_settings.game_mode == GameMode.Hockey
                    ), "Cannot use non-puck ball in hockey with sim"
                case BallTypeMutator.Basketball:
                    assert (
                        match_settings.game_mode == GameMode.Hoops
                    ), "Cannot use non-basketball ball in hoops with sim"
                case _:
                    raise NotImplementedError(mutators.ball_type)

            match mutators.boost:
                case BoostMutator.NormalBoost:
                    pass
                case BoostMutator.UnlimitedBoost:
                    mutator_config["boost_used_per_second"] = 0
                    mutator_config["car_spawn_boost_amount"] = 100
                case BoostMutator.NoBoost:
                    mutator_config["boost_accel"] = 0
                    print(
                        "Warning: No Boost boost option support is an experimental feature"
                    )
                case _:
                    raise NotImplementedError(mutators.boost)

            assert (
                mutators.rumble == RumbleMutator.NoRumble
            ), "Rumble is unsupported by sim"

            match mutators.boost_strength:
                case BoostStrengthMutator.One:
                    pass
                case BoostStrengthMutator.OneAndAHalf:
                    mutator_config["boost_accel"] = 21.2 * 1.5
                case BoostStrengthMutator.Two:
                    mutator_config["boost_accel"] = 21.2 * 2
                case BoostStrengthMutator.Five:
                    mutator_config["boost_accel"] = 21.2 * 5
                case BoostStrengthMutator.Ten:
                    mutator_config["boost_accel"] = 21.2 * 10

            match mutators.gravity:
                case GravityMutator.Default:
                    grav_z = -650
                case GravityMutator.Low:
                    grav_z = -325.0
                case GravityMutator.High:
                    grav_z = -1137.5
                case GravityMutator.SuperHigh:
                    grav_z = -3250
                case GravityMutator.Reverse:
                    grav_z = 650
            mutator_config["gravity"] = rsim.Vec(0, 0, grav_z)

            match mutators.demolish:
                case DemolishMutator.Default:
                    pass
                case DemolishMutator.Disabled:
                    mutator_config["demo_mode"] = rsim.DemoMode.DISABLED
                case DemolishMutator.OnContact:
                    mutator_config["demo_mode"] = rsim.DemoMode.ON_CONTACT
                case _:
                    raise NotImplementedError(mutators.demolish)

            match mutators.respawn_time:
                case RespawnTimeMutator.ThreeSeconds:
                    pass
                case RespawnTimeMutator.TwoSeconds:
                    mutator_config["respawn_delay"] = 2.0
                case RespawnTimeMutator.OneSecond:
                    mutator_config["respawn_delay"] = 1.0
                case _:
                    raise NotImplementedError(mutators.respawn_time)

            assert (
                mutators.game_event == GameEventMutator.Default
            ), f"game event option {mutators.game_event} is unsupported by sim"

            # TODO: BallMaxSpeedOption
            # TODO: BallWeightOption
            # TODO: BallSizeOption
            # TODO: BallBouncinessOption
        self._ball_touched_on_tick: Dict[int, bool] = {}
        self._touches: Dict[int, deque[int]] = {}
        self._car_id_spawn_id_map: Dict[int, int] = {}
        self._spawn_id_car_id_map: Dict[int, int] = {}
        self._current_car_ids: set[int] = set()
        self._tick_skip = tick_skip
        self._first_call = True
        self._tick_count = 0
        self._arena = rsim.Arena(mode)
        self._arena.set_ball_touch_callback(self._ball_touch_callback)
        if mutators:
            self._arena.set_mutator_config(rsim.MutatorConfig(**mutator_config))

    def _get_extra_ball_info(self) -> ExtraBallInfo:
        ball_state = self._arena.ball.get_state()
        return ExtraBallInfo(
            heatseeker_target_dir=ball_state.heatseeker_target_dir,
            heatseeker_target_speed=ball_state.heatseeker_target_speed,
            heatseeker_time_since_hit=ball_state.heatseeker_time_since_hit,
        )

    def _get_extra_player_info(self, car) -> ExtraPlayerInfo:
        car_state = car.get_state()
        return ExtraPlayerInfo(
            on_ground=car_state.is_on_ground,
            handbrake=car_state.handbrake_val,
            ball_touches=sum(self._touches[car.id]),
            car_contact_id=(
                0
                if car_state.car_contact_id == 0
                else self._car_id_spawn_id_map[car_state.car_contact_id]
            ),
            car_contact_cooldown_timer=car_state.car_contact_cooldown_timer,
            is_autoflipping=car_state.is_auto_flipping,
            autoflip_timer=car_state.auto_flip_timer,
            autoflip_direction=car_state.auto_flip_torque_scale,
        )

    def _get_extra_packet_info(self) -> ExtraPacketInfo:
        players = []
        for car in self._arena.get_cars():
            players.append(self._get_extra_player_info(car))
        return ExtraPacketInfo(players=players, ball=self._get_extra_ball_info())

    def get_extra_info(self, packet: GamePacket) -> ExtraPacketInfo:
        self._update_sim_cars(packet)
        if self._first_call:
            self._first_call = False
            self._tick_count = packet.match_info.frame_num
            self._set_sim_state(packet)
            return self._get_extra_packet_info()

        ticks_elapsed = packet.match_info.frame_num - self._tick_count
        self._tick_count = packet.match_info.frame_num
        spawn_id_player_info_map = {
            player_info.spawn_id: player_info for player_info in packet.players
        }
        for car in self._arena.get_cars():
            car_controls = rsim.CarControls()
            player_input = spawn_id_player_info_map[
                self._car_id_spawn_id_map[car.id]
            ].last_input
            car_controls.throttle = player_input.throttle
            car_controls.steer = player_input.steer
            car_controls.pitch = player_input.pitch
            car_controls.yaw = player_input.yaw
            car_controls.roll = player_input.roll
            car_controls.boost = player_input.boost
            car_controls.jump = player_input.jump
            car_controls.handbrake = player_input.handbrake
            car.set_controls(car_controls)
        for _ in range(ticks_elapsed):
            self._ball_touched_on_tick = {k: False for k in self._ball_touched_on_tick}
            self._arena.step(1)
            for k, v in self._ball_touched_on_tick.items():
                self._touches[k].append(v)
        self._set_sim_state(packet)
        return self._get_extra_packet_info()

    def _set_ball_state(self, packet: GamePacket):
        if len(packet.balls) > 0:
            ball = self._arena.ball
            ball_state = ball.get_state()
            packet_ball = packet.balls[0]
            packet_ball_physics = packet_ball.physics
            (latest_touch_player_idx, latest_touch_player_info) = max(
                enumerate(packet.players),
                key=lambda item: (
                    -1
                    if item[1].latest_touch is None
                    else item[1].latest_touch.game_seconds
                ),
            )
            if latest_touch_player_info.latest_touch is not None:
                ball_state.last_hit_car_id = self._spawn_id_car_id_map[
                    packet.players[latest_touch_player_idx].spawn_id
                ]
            ball_state.pos = rsim.Vec(
                packet_ball_physics.location.x,
                packet_ball_physics.location.y,
                packet_ball_physics.location.z,
            )
            ball_state.rot_mat = rsim.RotMat(
                *euler_to_rotation(rotator_to_numpy(packet_ball_physics.rotation))
                .transpose()
                .flatten()
            )
            ball_state.vel = rsim.Vec(*vector_to_numpy(packet_ball_physics.velocity))
            ball_state.ang_vel = rsim.Vec(
                *vector_to_numpy(packet_ball_physics.angular_velocity)
            )
            ball.set_state(ball_state)

    def _set_car_state(self, player_info: PlayerInfo, new: bool):
        if new:
            car = self._arena.add_car(
                player_info.team,
                Car.detect_hitbox(player_info.hitbox, player_info.hitbox_offset),
            )
        else:
            car = self._arena.get_car_from_id(
                self._spawn_id_car_id_map[player_info.spawn_id]
            )
        car_state = car.get_state()
        car_state.pos = rsim.Vec(*vector_to_numpy(player_info.physics.location))
        car_state.rot_mat = rsim.RotMat(
            *euler_to_rotation(rotator_to_numpy(player_info.physics.rotation))
            .transpose()
            .flatten()
        )
        car_state.vel = rsim.Vec(*vector_to_numpy(player_info.physics.velocity))
        car_state.ang_vel = rsim.Vec(
            *vector_to_numpy(player_info.physics.angular_velocity)
        )
        car_state.boost = player_info.boost
        car_state.is_supersonic = player_info.is_supersonic
        car_state.is_demoed = player_info.demolished_timeout != -1
        car_state.demo_respawn_timer = (
            player_info.demolished_timeout * car_state.is_demoed
        )
        car.set_state(car_state)
        self._touches = {
            car.id: deque([False] * self._tick_skip, self._tick_skip),
            **self._touches,
        }
        self._car_id_spawn_id_map[car.id] = player_info.spawn_id
        self._spawn_id_car_id_map[player_info.spawn_id] = car.id
        self._current_car_ids.add(car.id)

    def _is_new_car(self, player_info: PlayerInfo):
        return (
            player_info.spawn_id not in self._spawn_id_car_id_map
            or self._spawn_id_car_id_map[player_info.spawn_id]
            not in self._current_car_ids
        )

    def _update_sim_cars(self, packet: GamePacket):
        # Add data cars that are newly in the packet
        for player_info in packet.players:
            if self._is_new_car(player_info):
                self._set_car_state(player_info, True)

        # Remove data for cars that are no longer in the packet
        packet_car_ids = [
            self._spawn_id_car_id_map[player_info.spawn_id]
            for player_info in packet.players
        ]
        for car_id in list(self._current_car_ids):
            if car_id not in packet_car_ids:
                self._arena.remove_car(car_id)
                self._current_car_ids.remove(car_id)
                spawn_id = self._car_id_spawn_id_map.pop(car_id, None)
                if spawn_id is not None:
                    self._spawn_id_car_id_map.pop(spawn_id, None)
                self._touches.pop(car_id, None)
                self._ball_touched_on_tick.pop(car_id, None)

    def _set_sim_state(self, packet: GamePacket):
        for player_info in packet.players:
            if not self._is_new_car(player_info):
                self._set_car_state(player_info, False)
        self._set_ball_state(packet)

    def _ball_touch_callback(self, arena: rsim.Arena, car: rsim.Car, data):
        self._ball_touched_on_tick[car.id] = True
