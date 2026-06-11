import rclpy
import math
import json

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle

from smarc_msgs.action import BaseAction

from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union, triangulate
from smarc_action_base.gentler_action_server import GentlerActionServer


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
            ]
        )

        self.move_path_action = self.get_parameter('move_path_action').value
        self.lane_spacing     = self.get_parameter('lane_spacing').value
        self.waypoint_tol     = self.get_parameter('waypoint_tol').value
        self.speed            = self.get_parameter('speed').value

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

        self.get_logger().info('SearchArea (no obstacles, convex decomposition) started')


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
        payload   = self._pending_payload
        area_pts  = payload.get('area', [])
        self._speed_mission = payload.get('speed', self.speed)

        # Reset sub-goal state
        self._mp_goal_future   = None
        self._mp_result_future = None
        self._mp_goal_handle   = None
        self._last_feedback    = {}

        # Pre-compute waypoints (pure CPU — fine to do here)
        inside_latlon = [(p['lat'], p['lon']) for p in area_pts]
        self._waypoints_latlon = self._generate_coverage(inside_latlon, self.lane_spacing)

        if self._waypoints_latlon:
            self.get_logger().info(
                f'[SearchArea] Generated {len(self._waypoints_latlon)} waypoints'
            )
        else:
            self.get_logger().error('[SearchArea] No waypoints generated')

        # Check move_path server availability
        if not self._move_path_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'move_path action server unavailable: {self.move_path_action}')
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

            self.get_logger().info('[SearchArea] move_path accepted goal — waiting for result…')
            self._mp_result_future = self._mp_goal_handle.get_result_async()
            return None

        # ── Phase 3: wait for result ──────────────────────────────────────────
        if self._mp_result_future is not None:
            if not self._mp_result_future.done():
                return None

            result = self._mp_result_future.result()
            self._mp_result_future = None
            self._mp_goal_handle   = None

            if result.status == 4:   # SUCCEEDED
                self.get_logger().info('[SearchArea] move_path completed successfully')
                return True

            self.get_logger().error(f'[SearchArea] move_path ended with status={result.status}')
            return False

        return None  # fallback


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
    def _generate_coverage(self, inside_latlon, lane_spacing_m):
        origin_lat = sum(p[0] for p in inside_latlon) / len(inside_latlon)
        origin_lon = sum(p[1] for p in inside_latlon) / len(inside_latlon)

        inside_xy = [
            _latlon_to_xy(lat, lon, origin_lat, origin_lon)
            for lat, lon in inside_latlon
        ]

        poly = Polygon(inside_xy)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return []

        sweep_angle = self._longest_edge_angle(inside_xy)

        if self._is_convex(poly):
            self.get_logger().info('Zone is convex — direct sweep')
            ordered_xy = self._sweep_polygon(poly, lane_spacing_m, sweep_angle)
        else:
            self.get_logger().info('Zone is non-convex — triangular decomposition')
            cells = self._decompose_convex(poly)
            self.get_logger().info(f'Decomposed into {len(cells)} convex cell(s)')
            cell_paths = [
                wps for cell in cells
                if (wps := self._sweep_polygon(cell, lane_spacing_m, sweep_angle))
            ]
            if not cell_paths:
                return []
            ordered_xy = self._chain_paths(cell_paths)

        return [_xy_to_latlon(x, y, origin_lat, origin_lon) for x, y in ordered_xy]


    # ─────────────────────────────────────────────────────────────────────────
    def _is_convex(self, poly: Polygon) -> bool:
        hull = poly.convex_hull
        if hull.area < 1e-9:
            return True
        return abs(poly.area - hull.area) / hull.area < 0.01


    # ─────────────────────────────────────────────────────────────────────────
    def _decompose_convex(self, poly: Polygon) -> list:
        all_tris = triangulate(poly)
        tris = [t for t in all_tris if poly.contains(t) or poly.covers(t)]
        if not tris:
            return [poly]

        self.get_logger().info(f'  Triangulated into {len(tris)} triangle(s)')
        cells  = list(tris)
        merged = True

        while merged:
            merged    = False
            used      = [False] * len(cells)
            new_cells = []
            for i in range(len(cells)):
                if used[i]:
                    continue
                current = cells[i]
                for j in range(i + 1, len(cells)):
                    if used[j] or not current.touches(cells[j]):
                        continue
                    candidate = unary_union([current, cells[j]])
                    if (candidate.geom_type == 'Polygon'
                            and abs(candidate.area - candidate.convex_hull.area)
                            / (candidate.convex_hull.area + 1e-9) < 0.01):
                        current = candidate
                        used[j] = True
                        merged  = True
                used[i] = True
                new_cells.append(current)
            cells = new_cells

        return cells


    # ─────────────────────────────────────────────────────────────────────────
    def _sweep_polygon(self, poly: Polygon, lane_spacing_m: float,
                       sweep_angle: float) -> list:
        cos_a = math.cos(-sweep_angle)
        sin_a = math.sin(-sweep_angle)

        def rot(x, y):     return  cos_a*x - sin_a*y,  sin_a*x + cos_a*y
        def rot_inv(x, y): return  cos_a*x + sin_a*y, -sin_a*x + cos_a*y

        poly_rot = Polygon([rot(x, y) for x, y in poly.exterior.coords])
        minx, miny, maxx, maxy = poly_rot.bounds

        y       = maxy
        reverse = False
        waypoints_rot = []

        while y >= miny - lane_spacing_m * 0.5:
            line  = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
            inter = poly_rot.intersection(line)

            if not inter.is_empty:
                if inter.geom_type == 'LineString':
                    raw_segs = [inter]
                elif hasattr(inter, 'geoms'):
                    raw_segs = [g for g in inter.geoms if g.geom_type == 'LineString']
                else:
                    raw_segs = []

                if reverse:
                    raw_segs = list(reversed(raw_segs))

                for seg in raw_segs:
                    coords = list(seg.coords)
                    if len(coords) < 2:
                        continue
                    p1, p2 = coords[0], coords[-1]
                    if reverse:
                        p1, p2 = p2, p1
                    waypoints_rot.append(rot_inv(*p1))
                    waypoints_rot.append(rot_inv(*p2))

                reverse = not reverse

            y -= lane_spacing_m

        return waypoints_rot


    # ─────────────────────────────────────────────────────────────────────────
    def _chain_paths(self, cell_paths: list) -> list:
        remaining = list(cell_paths)
        result    = list(remaining.pop(0))

        while remaining:
            last      = result[-1]
            best_idx  = -1
            best_dist = float('inf')
            best_flip = False

            for i, path in enumerate(remaining):
                d_s = math.hypot(path[0][0]  - last[0], path[0][1]  - last[1])
                d_e = math.hypot(path[-1][0] - last[0], path[-1][1] - last[1])
                if d_s < best_dist:
                    best_dist, best_idx, best_flip = d_s, i, False
                if d_e < best_dist:
                    best_dist, best_idx, best_flip = d_e, i, True

            chosen = remaining.pop(best_idx)
            result.extend(reversed(chosen) if best_flip else chosen)

        return result


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