import rclpy
import math
import json

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle

from smarc_msgs.action import BaseAction

from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, triangulate
from shapely.affinity import affine_transform
from smarc_action_base.gentler_action_server import GentlerActionServer
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


_R_EARTH = 6371000.0

# ─────────────────────────────────────────────────────────────────────────
def _latlon_to_xy(lat, lon, origin_lat, origin_lon):
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    x = dlon * _R_EARTH * math.cos(math.radians(origin_lat))
    y = dlat * _R_EARTH
    return x, y


# ─────────────────────────────────────────────────────────────────────────
def _xy_to_latlon(x, y, origin_lat, origin_lon):
    dlat = y / _R_EARTH
    dlon = x / (_R_EARTH * math.cos(math.radians(origin_lat)))
    return origin_lat + math.degrees(dlat), origin_lon + math.degrees(dlon)


# ─────────────────────────────────────────────────────────────────────────
class SearchArea(Node):
    def __init__(self):
        super().__init__('search_area')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('move_path_action', rclpy.Parameter.Type.STRING),
                ('lane_spacing',     rclpy.Parameter.Type.DOUBLE),
                ('waypoint_tol',     rclpy.Parameter.Type.DOUBLE),
                ('speed',            rclpy.Parameter.Type.DOUBLE),
                ('obstacle_buffer',  rclpy.Parameter.Type.DOUBLE),
            ]
        )

        self.move_path_action = self.get_parameter('move_path_action').value
        self.lane_spacing     = self.get_parameter('lane_spacing').value
        self.waypoint_tol     = self.get_parameter('waypoint_tol').value
        self.speed            = self.get_parameter('speed').value
        self.obstacle_buffer  = self.get_parameter('obstacle_buffer').value

        cbg = ReentrantCallbackGroup()

        self._move_path_client = ActionClient(
            self, BaseAction, self.move_path_action, callback_group=cbg)

        # ── Mission state ─────────────────────────────────────────────────────
        self._pending_payload: dict = {}
        self._waypoints_latlon: list = []
        self._speed_mission: float = self.speed

        # ── move_path sub-goal state ──────────────────────────────────────────
        self._mp_goal_future   = None
        self._mp_result_future = None
        self._mp_goal_handle: ClientGoalHandle | None = None
        self._last_feedback: dict = {}

        # ── GentlerActionServer ───────────────────────────────────────────────
        self._server = GentlerActionServer(
            node=self,
            action_name='search_area',
            on_goal_received=self._on_goal_received,
            on_cancel_received=self._on_cancel_received,
            prepare_loop=self._prepare_loop,
            loop_inner=self._loop_inner,
            give_feedback=self._give_feedback,
            loop_frequency=5.0,
        )
        
        self.get_logger().info('SearchArea (polygon-with-holes sweep) started')


    # ─────────────────────────────────────────────────────────────────────────
    def _on_goal_received(self, payload: dict) -> bool:
        area_pts = payload.get('area', [])
        if not area_pts or len(area_pts) < 3:
            self.get_logger().error('[Goal] Need >= 3 area points')
            return False
        self._pending_payload = payload
        return True


    # ─────────────────────────────────────────────────────────────────────────
    def _on_cancel_received(self) -> bool:
        self.get_logger().info('[SearchArea] Cancel requested')
        if self._mp_goal_handle is not None:
            self.get_logger().info('[SearchArea] Cancelling active move_path sub-goal')
            self._mp_goal_handle.cancel_goal_async()
        return True
        

    # ─────────────────────────────────────────────────────────────────────────
    def _prepare_loop(self) -> None:
        payload       = self._pending_payload
        area_pts      = payload.get('area', [])
        obstacles_raw = payload.get('obstacles', [])
        self._speed_mission = payload.get('speed', self.speed)
        buf_dist      = payload.get('obstacle_buffer', self.obstacle_buffer)

        # Reset sub-goal state
        self._mp_goal_future   = None
        self._mp_result_future = None
        self._mp_goal_handle   = None
        self._last_feedback    = {}

        inside_latlon    = [(p['lat'], p['lon']) for p in area_pts]
        obstacles_latlon = [
            [(p['lat'], p['lon']) for p in obs]
            for obs in obstacles_raw
            if len(obs) >= 3
        ]

        self._waypoints_latlon = self._generate_coverage(
            inside_latlon, obstacles_latlon, self.lane_spacing, buf_dist)

        if self._waypoints_latlon:
            self.get_logger().info(
                f'[SearchArea] Generated {len(self._waypoints_latlon)} waypoints'
            )
        else:
            self.get_logger().error('[SearchArea] No waypoints generated')
        if not self._move_path_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'move_path action server unavailable: {self.move_path_action}')
            self._server_unavailable = True
        else:
            self._server_unavailable = False


    # ─────────────────────────────────────────────────────────────────────────
    def _loop_inner(self) -> bool | None:
        if getattr(self, '_server_unavailable', False):
            return False

        if not self._waypoints_latlon:
            return False

        # ── Phase 1: send goal ────────────────────────────────────────────────
        if self._mp_goal_future is None and self._mp_result_future is None:
            goal_msg = BaseAction.Goal()
            goal_msg.goal.data = json.dumps({
                'speed': self._speed_mission,
                'waypoints': [
                    {'latitude': lat, 'longitude': lon, 'tolerance': self.waypoint_tol}
                    for lat, lon in self._waypoints_latlon
                ],
            })
            self.get_logger().info(
                f'[SearchArea] Sending {len(self._waypoints_latlon)} waypoints to move_path'
            )
            self._mp_goal_future = self._move_path_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_cb,
            )
            return None

        # ── Phase 2: wait for acceptance ──────────────────────────────────────
        if self._mp_goal_future is not None and self._mp_result_future is None:
            if not self._mp_goal_future.done():
                return None

            self._mp_goal_handle = self._mp_goal_future.result()
            self._mp_goal_future = None

            if not self._mp_goal_handle.accepted:
                self.get_logger().error('[SearchArea] move_path rejected goal')
                return False

            self.get_logger().info(
                '[SearchArea] move_path accepted goal — waiting for result…')
            self._mp_result_future = self._mp_goal_handle.get_result_async()
            return None

        # ── Phase 3: wait for result ──────────────────────────────────────────
        if self._mp_result_future is not None:
            if not self._mp_result_future.done():
                return None

            result = self._mp_result_future.result()
            self._mp_result_future = None
            self._mp_goal_handle   = None

            if result.status == 4:
                self.get_logger().info('[SearchArea] move_path completed successfully')
                return True

            self.get_logger().error(
                f'[SearchArea] move_path ended with status={result.status}')
            return False

        return None


    # ─────────────────────────────────────────────────────────────────────────
    def _give_feedback(self) -> str:
        return json.dumps({
            'waypoints_total': len(self._waypoints_latlon),
            'sub_feedback':    self._last_feedback,
        })


    # ─────────────────────────────────────────────────────────────────────────
    def _feedback_cb(self, feedback_msg):
        try:
            data = json.loads(feedback_msg.feedback.feedback.data)
            self._last_feedback = data
            self.get_logger().info(
                f'[move_path] '
                f'total={data.get("total_progress_pct", "?")}% | '
                f'wp[{data.get("wp_current_idx", "?")}]={data.get("wp_progress_pct", "?")}% | '
                f'precision={data.get("precision_pct", "?")}% | '
                f'distance={data.get("distance_travelled", "?")}m'
            )
        except Exception:
            pass


    # ─────────────────────────────────────────────────────────────────────────
    def _generate_coverage(self, inside_latlon, obstacles_latlon,
                           lane_spacing_m, buf_dist_m):
        origin_lat = sum(p[0] for p in inside_latlon) / len(inside_latlon)
        origin_lon = sum(p[1] for p in inside_latlon) / len(inside_latlon)

        def to_xy(latlon_list):
            return [_latlon_to_xy(lat, lon, origin_lat, origin_lon)
                    for lat, lon in latlon_list]

        area_poly = Polygon(to_xy(inside_latlon))
        if not area_poly.is_valid:
            area_poly = area_poly.buffer(0)

        buffered_obstacles = []
        for obs_latlon in obstacles_latlon:
            obs_poly = Polygon(to_xy(obs_latlon))
            if not obs_poly.is_valid:
                obs_poly = obs_poly.buffer(0)
            clipped = obs_poly.buffer(buf_dist_m).intersection(area_poly)
            if not clipped.is_empty:
                buffered_obstacles.append(clipped)

        if buffered_obstacles:
            search_zone = area_poly.difference(unary_union(buffered_obstacles))
        else:
            search_zone = area_poly

        if search_zone.is_empty:
            self.get_logger().error('Search zone is empty after obstacle subtraction')
            return []

        self.get_logger().info(
            f'Search zone: {search_zone.geom_type} | '
            f'{len(buffered_obstacles)} hole(s) carved out'
        )

        sweep_angle = self._longest_edge_angle(to_xy(inside_latlon))
        ordered_xy  = self._sweep_holed_polygon(
            search_zone, lane_spacing_m, sweep_angle)  

        self._search_zone_xy    = search_zone
        self._origin_lat        = origin_lat
        self._origin_lon        = origin_lon

        if not ordered_xy:
            return []

        return [_xy_to_latlon(x, y, origin_lat, origin_lon) for x, y in ordered_xy]


    # ─────────────────────────────────────────────────────────────────────────
    def _sweep_holed_polygon(self, geom, lane_spacing_m: float,
                             sweep_angle: float) -> list:
        cos_a = math.cos(-sweep_angle)
        sin_a = math.sin(-sweep_angle)

        def rot_inv(x, y):
            return cos_a*x + sin_a*y, -sin_a*x + cos_a*y

        geom_rot = affine_transform(geom, [cos_a, -sin_a, sin_a, cos_a, 0, 0])
        minx, miny, maxx, maxy = geom_rot.bounds

        y       = maxy
        reverse = False

        left_channel_segments  = []
        right_channel_segments = []
        waypoints_final        = []

        while y >= miny - lane_spacing_m * 0.5:
            line  = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
            inter = geom_rot.intersection(line)

            if not inter.is_empty:
                if inter.geom_type == 'LineString':
                    raw_segs = [inter]
                elif inter.geom_type in ('MultiLineString', 'GeometryCollection'):
                    raw_segs = [g for g in inter.geoms
                                if g.geom_type == 'LineString']
                else:
                    raw_segs = []

                if raw_segs:
                    raw_segs.sort(
                        key=lambda s: (s.coords[0][0] + s.coords[-1][0]) / 2)

                    if len(raw_segs) == 1:
                        if left_channel_segments or right_channel_segments:
                            waypoints_final.extend(
                                self._resolve_channels(
                                    left_channel_segments,
                                    right_channel_segments,
                                    rot_inv,
                                )
                            )
                            left_channel_segments  = []
                            right_channel_segments = []

                        seg = raw_segs[0]
                        p1, p2 = (
                            (seg.coords[-1], seg.coords[0]) if reverse
                            else (seg.coords[0], seg.coords[-1])
                        )
                        waypoints_final.append(rot_inv(*p1))
                        waypoints_final.append(rot_inv(*p2))
                        reverse = not reverse

                    else:
                        left_channel_segments.append((raw_segs[0],  reverse))
                        right_channel_segments.append((raw_segs[-1], reverse))
                        reverse = not reverse

            y -= lane_spacing_m

        if left_channel_segments or right_channel_segments:
            waypoints_final.extend(
                self._resolve_channels(
                    left_channel_segments, right_channel_segments, rot_inv)
            )

        return waypoints_final


    # ─────────────────────────────────────────────────────────────────────────
    def _resolve_channels(self, left_chan, right_chan, rot_inv_func):
        local_wps = []

        for seg, rev in left_chan:
            p1, p2 = (
                (seg.coords[-1], seg.coords[0]) if rev
                else (seg.coords[0], seg.coords[-1])
            )
            local_wps.append(rot_inv_func(*p1))
            local_wps.append(rot_inv_func(*p2))

        for seg, rev in reversed(right_chan):
            p1, p2 = (
                (seg.coords[0], seg.coords[-1]) if rev
                else (seg.coords[-1], seg.coords[0])
            )
            local_wps.append(rot_inv_func(*p1))
            local_wps.append(rot_inv_func(*p2))

        return self._fix_crossings(local_wps)


    # ─────────────────────────────────────────────────────────────────────────
    def _fix_crossings(self, wps: list) -> list:

        if not hasattr(self, '_search_zone_xy') or len(wps) < 2:
            return wps

        from shapely.geometry import LineString as SLS, Point as SPoint

        result = [wps[0]]

        for i in range(len(wps) - 1):
            a = wps[i]
            b = wps[i + 1]
            seg = SLS([a, b])

            if self._search_zone_xy.contains(seg):
                # Transition légale, on garde
                result.append(b)
            else:
                # Transition illégale : longer le bord de la search_zone
                detour = self._boundary_detour(a, b)
                result.extend(detour)

        return result


    # ─────────────────────────────────────────────────────────────────────────
    def _boundary_detour(self, a: tuple, b: tuple) -> list:
        from shapely.geometry import Point as SPoint
        from shapely.ops import nearest_points

        boundary = self._search_zone_xy.boundary

        pa = nearest_points(boundary, SPoint(a))[0]
        pb = nearest_points(boundary, SPoint(b))[0]

        try:
            coords = list(boundary.coords)
        except NotImplementedError:

            coords = list(self._search_zone_xy.exterior.coords)

        def nearest_idx(pt):
            min_d, idx = float('inf'), 0
            for k, c in enumerate(coords):
                d = (c[0] - pt.x)**2 + (c[1] - pt.y)**2
                if d < min_d:
                    min_d, idx = d, k
            return idx

        ia = nearest_idx(pa)
        ib = nearest_idx(pb)
        n  = len(coords)

        if ia <= ib:
            path_fwd = coords[ia:ib + 1]
            path_rev = coords[ib:] + coords[:ia + 1]
        else:
            path_fwd = coords[ia:] + coords[:ib + 1]
            path_rev = coords[ib:ia + 1]

        def path_len(pts):
            return sum(math.hypot(pts[k+1][0]-pts[k][0], pts[k+1][1]-pts[k][1])
                    for k in range(len(pts)-1))

        chosen = path_fwd if path_len(path_fwd) <= path_len(path_rev) else path_rev

        detour = [(c[0], c[1]) for c in chosen]
        detour.append(b)
        return detour


    # ─────────────────────────────────────────────────────────────────────────
    def _longest_edge_angle(self, pts: list) -> float:
        best_len, best_angle = -1.0, 0.0
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            l = math.hypot(x2 - x1, y2 - y1)
            if l > best_len:
                best_len   = l
                best_angle = math.atan2(y2 - y1, x2 - x1)
        return best_angle


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = SearchArea()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()