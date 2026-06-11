import rclpy
import json

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle

from smarc_msgs.action import BaseAction
from smarc_msgs.msg import GeofencePolygonsStamped
from smarc_utilities import georef_utils
from geographic_msgs.msg import GeoPoint
from smarc_msgs.msg import Topics as smarcTopics
from smarc_action_base.gentler_action_server import GentlerActionServer
from geometry_msgs.msg import PointStamped

# ─────────────────────────────────────────────────────────────────────────
class SearchAreas(Node):

    _SPEED_MAP = {'high': 10.0, 'medium': 5.0, 'low': 2.0}

    def __init__(self):
        super().__init__('search_areas')

        cbg = ReentrantCallbackGroup()

        self._obstacles_latlon: list[list[dict]] = []
        self._geofence_received: bool = False

        self._geofence_sub = self.create_subscription(GeofencePolygonsStamped, smarcTopics.GEOFENCE_POLYGONS_TOPIC, self._geofence_cb, 10, callback_group=cbg,)

        self._search_area_client = ActionClient(self, BaseAction, 'search_area', callback_group=cbg,)

        # ── Mission state (populated in on_goal_received / prepare_loop) ──────
        self._areas: list = []
        self._speed: float = 5.0
        self._obstacles_snapshot: list[list[dict]] = []

        # Per-iteration state (managed inside loop_inner)
        self._current_area_index: int = 0
        self._current_goal_handle: ClientGoalHandle | None = None
        self._area_future = None          
        self._result_future = None        
        self._last_feedback: dict = {}

        # ── GentlerActionServer ───────────────────────────────────────────────
        self._server = GentlerActionServer(
            node=self,
            action_name='search_areas',
            on_goal_received=self._on_goal_received,
            on_cancel_received=self._on_cancel_received,
            prepare_loop=self._prepare_loop,
            loop_inner=self._loop_inner,
            give_feedback=self._give_feedback,
            loop_frequency=5.0,
        )

        self.get_logger().info('SearchAreas orchestrator started')
        self.get_logger().info(
            f'Waiting for goals on: {self.get_namespace()}/search_areas'
        )


    # ─────────────────────────────────────────────────────────────────────────
    def _geofence_cb(self, msg: GeofencePolygonsStamped):
        obstacles: list[list[dict]] = []

        for poly in msg.islands:
            pts: list[dict] = []
            for pt in poly.points:
                try:
                    pt_stamped = PointStamped()
                    pt_stamped.header.frame_id = "utm_33_V"
                    pt_stamped.header.stamp = msg.header.stamp 

                    pt_stamped.point.x = float(pt.x)
                    pt_stamped.point.y = float(pt.y)
                    pt_stamped.point.z = float(pt.z)

                    gp: GeoPoint = georef_utils.convert_utm_to_latlon(pt_stamped)
                    
                    pts.append({'lat': gp.latitude, 'lon': gp.longitude})
                except Exception as e:
                    self.get_logger().warn(
                        f'[Geofence] Could not convert point {pt}: {e}'
                    )

            if len(pts) >= 3:
                obstacles.append(pts)
            else:
                self.get_logger().warn(
                    '[Geofence] Dropped island polygon with < 3 valid points'
                )

        self._obstacles_latlon = obstacles
        self._geofence_received = True
        self.get_logger().info(
            f'[Geofence] Updated: {len(obstacles)} obstacle(s) cached'
        )


    # ─────────────────────────────────────────────────────────────────────────
    def _resolve_obstacles(self, payload: dict) -> list[list[dict]]:
        if self._geofence_received:
            self.get_logger().info(
                f'[Obstacles] Using {len(self._obstacles_latlon)} '
                'obstacle(s) from /smarc/geofence_polygons'
            )
            return list(self._obstacles_latlon)

        polygones = payload.get('polygones', [])
        obstacles = [
            [{'lat': pt['lat'], 'lon': pt['lon']} for pt in poly['points']]
            for poly in polygones
            if not poly.get('stay_inside', True)
            and len(poly.get('points', [])) >= 3
        ]

        if obstacles:
            self.get_logger().warn(
                f'[Obstacles] No geofence topic received — '
                f'using {len(obstacles)} obstacle(s) from JSON payload'
            )
        else:
            self.get_logger().warn(
                '[Obstacles] No geofence topic and no polygones in payload — '
                'proceeding with zero obstacles'
            )
        return obstacles


    # ─────────────────────────────────────────────────────────────────────────
    def _on_goal_received(self, payload: dict) -> bool:
        try:
            params = payload['task']['params']
            areas  = params['areas']
        except (KeyError, TypeError) as e:
            self.get_logger().error(f'[Goal] Missing task/params/areas: {e}')
            return False

        if not areas:
            self.get_logger().error('[Goal] Empty areas list')
            return False

        self._pending_payload = payload
        return True


    # ─────────────────────────────────────────────────────────────────────────
    def _on_cancel_received(self) -> bool:
        self.get_logger().info('[SearchAreas] Cancel requested')
        if self._current_goal_handle is not None:
            self.get_logger().info('[SearchAreas] Cancelling active search_area sub-goal')
            self._current_goal_handle.cancel_goal_async()

        return True


    # ─────────────────────────────────────────────────────────────────────────
    def _prepare_loop(self) -> None:
        """Called once before loop_inner starts. Parse payload, reset state."""
        payload   = self._pending_payload
        params    = payload['task']['params']
        speed_raw = params.get('speed', 'low')

        self._areas = params['areas']
        self._speed = (
            self._SPEED_MAP.get(speed_raw.lower(), 5.0)
            if isinstance(speed_raw, str)
            else float(speed_raw)
        )
        self._obstacles_snapshot = self._resolve_obstacles(payload)

        self._current_area_index = 0
        self._current_goal_handle = None
        self._area_future  = None
        self._result_future = None
        self._last_feedback = {}

        valid = []
        for i, pts in enumerate(self._areas):
            if pts and len(pts) >= 3:
                valid.append(pts)
            else:
                self.get_logger().warn(
                    f'[SearchAreas] Area {i + 1} has < 3 points — skipping'
                )
        self._areas = valid

        self.get_logger().info(
            f'[SearchAreas] Mission ready: {len(self._areas)} area(s) | '
            f'speed={self._speed} | {len(self._obstacles_snapshot)} obstacle(s)'
        )

        if not self._search_area_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('search_area action server unavailable')
            self._server_unavailable = True
        else:
            self._server_unavailable = False


    # ─────────────────────────────────────────────────────────────────────────
    def _loop_inner(self) -> bool | None:

        if getattr(self, '_server_unavailable', False):
            return False

        if self._current_area_index >= len(self._areas):
            self.get_logger().info('[SearchAreas] All areas completed')
            return True

        area_pts  = self._areas[self._current_area_index]
        zone_name = f'Area {self._current_area_index + 1}'

        # ── Phase 1: send the goal (once) ─────────────────────────────────────
        if self._area_future is None and self._result_future is None:
            area_latlon = [
                {'lat': pt['latitude'], 'lon': pt['longitude']}
                for pt in area_pts
            ]
            zone_payload = {
                'speed':     self._speed,
                'area':      area_latlon,
                'obstacles': self._obstacles_snapshot,
            }
            goal_msg           = BaseAction.Goal()
            goal_msg.goal.data = json.dumps(zone_payload)

            self.get_logger().info(
                f'[SearchAreas] Sending {zone_name} to search_area '
                f'({len(area_latlon)} pts, {len(self._obstacles_snapshot)} obstacle(s))'
            )
            self._area_future = self._search_area_client.send_goal_async(
                goal_msg,
                feedback_callback=self._search_area_feedback_cb,
            )
            return None 

        # ── Phase 2: goal sent — wait for acceptance ──────────────────────────
        if self._area_future is not None and self._result_future is None:
            if not self._area_future.done():
                return None 

            self._current_goal_handle = self._area_future.result()
            self._area_future = None

            if not self._current_goal_handle.accepted:
                self.get_logger().error(
                    f'[SearchAreas] search_area rejected {zone_name}'
                )
                return False

            self.get_logger().info(
                f'[SearchAreas] search_area accepted {zone_name} — waiting for result…'
            )
            self._result_future = self._current_goal_handle.get_result_async()
            return None

        # ── Phase 3: goal accepted — wait for result ──────────────────────────
        if self._result_future is not None:
            if not self._result_future.done():
                return None  

            result = self._result_future.result()
            self._result_future = None
            self._current_goal_handle = None

            if result.status == 4:
                self.get_logger().info(
                    f'[SearchAreas] {zone_name} completed '
                    f'({self._current_area_index + 1}/{len(self._areas)})'
                )
                self._current_area_index += 1
                return None 
            else:
                self.get_logger().error(
                    f'[SearchAreas] {zone_name} failed with status={result.status}'
                )
                return False

        return None  # fallback


    # ─────────────────────────────────────────────────────────────────────────
    def _give_feedback(self) -> str:
        """Return a JSON feedback string for the outer client."""
        zone_name = (
            f'Area {self._current_area_index + 1}'
            if self._current_area_index < len(self._areas)
            else 'done'
        )
        return json.dumps({
            'zones_done':   self._current_area_index,
            'zones_total':  len(self._areas),
            'current_zone': zone_name,
            'sub_feedback': self._last_feedback,
        })


    # ─────────────────────────────────────────────────────────────────────────
    def _search_area_feedback_cb(self, feedback_msg):
        try:
            self._last_feedback = json.loads(
                feedback_msg.feedback.feedback.data
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = SearchAreas()

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