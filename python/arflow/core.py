"""Data exchanging service."""

import os
import pickle
import re
import struct
import time
import uuid
from collections import namedtuple
from pathlib import Path
from time import gmtime, strftime

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import log_pinhole

from arflow import service_pb2, service_pb2_grpc

sessions: dict[str, service_pb2.RegisterRequest] = {}
"""@private"""


HandData = namedtuple(
    "HandData",
    [
        "left_tracked",
        "left_points",
        "left_confidence",
        "right_tracked",
        "right_points",
        "right_confidence",
        "timestamp",
    ],
)

META_QUEST_PALM_CONNECTIONS = ((0, 1), (1, 2), (1, 5), (1, 8), (1, 11), (1, 14))
META_QUEST_THUMB_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 18))
META_QUEST_INDEX_FINGER_CONNECTIONS = ((0, 1), (1, 5), (5, 6), (6, 7), (7, 19))
META_QUEST_MIDDLE_FINGER_CONNECTIONS = ((0, 1), (1, 8), (8, 9), (9, 10), (10, 20))
META_QUEST_RING_FINGER_CONNECTIONS = ((0, 1), (1, 11), (11, 12), (12, 13), (13, 21))
META_QUEST_PINKY_FINGER_CONNECTIONS = ((0, 1), (1, 14), (14, 15), (15, 16), (16, 17), (17, 22))
META_QUEST_LINKS: tuple[tuple[int, int], ...] = (
    # META_QUEST_PALM_CONNECTIONS
    META_QUEST_THUMB_CONNECTIONS
    + META_QUEST_INDEX_FINGER_CONNECTIONS
    + META_QUEST_MIDDLE_FINGER_CONNECTIONS
    + META_QUEST_RING_FINGER_CONNECTIONS
    + META_QUEST_PINKY_FINGER_CONNECTIONS
)

META_QUEST_ID2NAME: dict[int, str] = {
    0: "forearm_stub",  # Hand_ForearmStub   (virtual, a few cm up the forearm)
    1: "wrist_root",  # Hand_WristRoot
    # ――― Thumb (metacarpal + 3 phalanges) ―――
    2: "thumb_1",  # Hand_Thumb1  – proximal / MCP
    3: "thumb_2",  # Hand_Thumb2  – intermediate / IP
    4: "thumb_3",  # Hand_Thumb3  – distal
    # ――― Index finger (no "0" bone in the enum) ―――
    5: "index_1",  # Hand_Index1  – metacarpal / MCP
    6: "index_2",  # Hand_Index2  – PIP
    7: "index_3",  # Hand_Index3  – DIP
    # ――― Middle finger ―――
    8: "middle_1",  # Hand_Middle1 – metacarpal / MCP
    9: "middle_2",  # Hand_Middle2 – PIP
    10: "middle_3",  # Hand_Middle3 – DIP
    # ――― Ring finger ―――
    11: "ring_1",  # Hand_Ring1
    12: "ring_2",  # Hand_Ring2
    13: "ring_3",  # Hand_Ring3
    # ――― Pinky (has a "0" metacarpal) ―――
    14: "pinky_0",  # Hand_Pinky0 – metacarpal
    15: "pinky_1",  # Hand_Pinky1 – MCP
    16: "pinky_2",  # Hand_Pinky2 – PIP
    17: "pinky_3",  # Hand_Pinky3 – DIP
    18: "thumb_tip",  # Hand_ThumbTip
    19: "index_tip",  # Hand_IndexTip
    20: "middle_tip",  # Hand_MiddleTip
    21: "ring_tip",  # Hand_RingTip
    22: "pinky_tip",  # Hand_PinkyTip
}

META_QUEST_IDS: list[int] = [int(key) for key in META_QUEST_ID2NAME]


def set_pose_annotation_context(parent_log_path: Path) -> None:
    rr.log(
        f"{parent_log_path}",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Meta Quest Keypoints", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in META_QUEST_ID2NAME.items()
                    ],
                    keypoint_connections=META_QUEST_LINKS,
                ),
            ]
        ),
        static=True,
    )


class ARFlowService(service_pb2_grpc.ARFlowService):
    """ARFlow gRPC service."""

    _start_time = time.time_ns()
    _frame_data: list[dict[str, float | bytes]] = []

    def __init__(self) -> None:
        super().__init__()

    def _save_frame_data(self, request: service_pb2.DataFrameRequest | service_pb2.RegisterRequest):
        """@private"""
        time_stamp = (time.time_ns() - self._start_time) / 1e9
        self._frame_data.append({"time_stamp": time_stamp, "data": request.SerializeToString()})

    def register(
        self, request: service_pb2.RegisterRequest, context, uid: str | None = None
    ) -> service_pb2.RegisterResponse:
        """Register a client."""

        self._save_frame_data(request)

        # Start processing.
        if uid is None:
            uid = str(uuid.uuid4())

        sessions[uid] = request

        rr.init(f"{request.device_name} - ARFlow", spawn=True)
        self.parent_log_path = Path("world")
        print(f"Registered a client with UUID: {uid}", request)

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(),
                rrb.Grid(
                    rrb.Spatial2DView(origin=f"{self.parent_log_path}/camera/pinhole/rgb"),
                    rrb.Spatial2DView(origin=f"{self.parent_log_path}/camera/pinhole/depth"),
                ),
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint=blueprint)

        rr.log("/", rr.ViewCoordinates.RDF, static=True)
        set_pose_annotation_context(self.parent_log_path)

        # Call the for user extension code.
        self.on_register(request)

        return service_pb2.RegisterResponse(uid=uid)

    def data_frame(
        self,
        request: service_pb2.DataFrameRequest,
        context,
    ) -> service_pb2.DataFrameResponse:
        """Process an incoming frame."""

        self._save_frame_data(request)

        # Start processing.
        decoded_data = {}
        session_configs = sessions[request.uid]

        cam_log_path = self.parent_log_path / "camera"
        pinhole_log_path: Path = cam_log_path / "pinhole"
        if session_configs.camera_color.enabled:
            color_rgb = ARFlowService.decode_rgb_image(session_configs, request.color)
            decoded_data["color_rgb"] = color_rgb
            color_rgb = np.fliplr(color_rgb)
            rr.log(f"{pinhole_log_path}/rgb", rr.Image(color_rgb).compress(jpeg_quality=80))

        if session_configs.camera_depth.enabled:
            depth_img = ARFlowService.decode_depth_image(session_configs, request.depth)
            decoded_data["depth_img"] = depth_img
            depth_img = np.fliplr(depth_img)
            rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(depth_img, meter=1.0))
        else:
            print("camera_depth.enabled else")
            hands_data: HandData = self.deserialize_hand_tracking_data(request.depth)
            decoded_data["hands"] = hands_data
            if hands_data.left_tracked:
                left_xyz = np.array(hands_data.left_points, dtype=np.float32)
                # remove the first point, which is some weird extra point
                left_xyz = left_xyz[1:]

                rr.log(
                    f"{self.parent_log_path}/left_hand",
                    rr.Points3D(
                        left_xyz,
                        class_ids=0,
                        keypoint_ids=META_QUEST_IDS,
                        show_labels=False,
                        # colors=np.full((len(hands_data.left_points), 3), 255, dtype=np.uint8),
                    ),
                )
            if hands_data.right_tracked:
                right_xyz = np.array(hands_data.right_points, dtype=np.float32)
                # remove the first point, which is some weird extra point
                right_xyz = right_xyz[1:]
                rr.log(
                    f"{self.parent_log_path}/right_hand",
                    rr.Points3D(
                        right_xyz,
                        class_ids=0,
                        keypoint_ids=META_QUEST_IDS,
                        show_labels=False,
                        # colors=np.full((len(hands_data.right_points), 3), 255, dtype=np.uint8),
                    ),
                )

        if session_configs.camera_transform.enabled:
            # rr.log(f"{self.parent_log_path}", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

            transform = ARFlowService.decode_transform(request.transform)
            decoded_data["transform"] = transform

            ulf_to_rdf = np.array(
                [
                    [0.0, -1.0, 0.0, 0],
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            world_T_cam = transform @ ulf_to_rdf

            extri = Extrinsics(world_R_cam=world_T_cam[:3, :3], world_t_cam=world_T_cam[:3, 3])

            k = ARFlowService.decode_intrinsic(session_configs)
            intri = Intrinsics(
                camera_conventions="RDF",
                fl_x=k[0, 0],
                fl_y=k[1, 1],
                cx=k[0, 2],
                cy=k[1, 2],
                height=color_rgb.shape[0],
                width=color_rgb.shape[1],
            )
            pinhole_param = PinholeParameters(name="camera", intrinsics=intri, extrinsics=extri)
            log_pinhole(pinhole_param, cam_log_path=cam_log_path, image_plane_distance=0.1)

        # if session_configs.camera_point_cloud.enabled:
        #     pcd, clr = ARFlowService.decode_point_cloud(session_configs, k, color_rgb, depth_img, transform)
        #     decoded_data["point_cloud_pcd"] = pcd
        #     decoded_data["point_cloud_clr"] = clr
        #     rr.log("world/point_cloud", rr.Points3D(pcd, colors=clr))

        # Call the for user extension code.
        self.on_frame_received(decoded_data)

        return service_pb2.DataFrameResponse(message="OK")

    def on_register(self, request: service_pb2.RegisterRequest):
        """Called when a new device is registered. Override this method to process the data."""
        pass

    def on_frame_received(self, frame_data: service_pb2.DataFrameRequest):
        """Called when a frame is received. Override this method to process the data."""
        pass

    def on_program_exit(self, path_to_save: str | None):
        """Save the data and exit."""
        if path_to_save is None:
            return
        print("Saving the data...")
        f_name = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        save_path = os.path.join(path_to_save, f"frames_{f_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self._frame_data, f)

        print(f"Data saved to {save_path}")

    @staticmethod
    def decode_rgb_image(session_configs: service_pb2.RegisterRequest, buffer: bytes) -> np.ndarray:
        # Calculate the size of the image.
        color_img_w = int(session_configs.camera_intrinsics.resolution_x * session_configs.camera_color.resize_factor_x)
        color_img_h = int(session_configs.camera_intrinsics.resolution_y * session_configs.camera_color.resize_factor_y)
        p = color_img_w * color_img_h
        color_img = np.frombuffer(buffer, dtype=np.uint8)

        # Decode RGB bytes into RGB.
        if session_configs.camera_color.data_type == "RGB24":
            color_rgb = color_img.reshape((color_img_h, color_img_w, 3))
            color_rgb = color_rgb.astype(np.uint8)

        # Decode YCbCr bytes into RGB.
        elif session_configs.camera_color.data_type == "YCbCr420":
            y = color_img[:p].reshape((color_img_h, color_img_w))
            cbcr = color_img[p:].reshape((color_img_h // 2, color_img_w // 2, 2))
            cb, cr = cbcr[:, :, 0], cbcr[:, :, 1]

            # Very important! Convert to float32 first!
            cb = np.repeat(cb, 2, axis=0).repeat(2, axis=1).astype(np.float32) - 128
            cr = np.repeat(cr, 2, axis=0).repeat(2, axis=1).astype(np.float32) - 128

            r = np.clip(y + 1.403 * cr, 0, 255)
            g = np.clip(y - 0.344 * cb - 0.714 * cr, 0, 255)
            b = np.clip(y + 1.772 * cb, 0, 255)

            color_rgb = np.stack([r, g, b], axis=-1)
            color_rgb = color_rgb.astype(np.uint8)

        return color_rgb

    @staticmethod
    def decode_depth_image(session_configs: service_pb2.RegisterRequest, buffer: bytes) -> np.ndarray:
        if session_configs.camera_depth.data_type == "f32":
            dtype = np.float32
        elif session_configs.camera_depth.data_type == "u16":
            dtype = np.uint16
        else:
            raise ValueError(f"Unknown depth data type: {session_configs.camera_depth.data_type}")

        depth_img = np.frombuffer(buffer, dtype=dtype)
        depth_img = depth_img.reshape(
            (
                session_configs.camera_depth.resolution_y,
                session_configs.camera_depth.resolution_x,
            )
        )

        # 16-bit unsigned integer, describing the depth (distance to an object) in millimeters.
        if dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) / 1000.0

        return depth_img

    @staticmethod
    def decode_transform(buffer: bytes):
        y_down_to_y_up = np.array(
            [
                [1.0, 0.0, 0.0, 0],
                [0.0, -1.0, 0.0, 0],
                [0.0, 0.0, 1.0, 0],
                [0.0, 0.0, 0, 1.0],
            ],
            dtype=np.float32,
        )

        t = np.frombuffer(buffer, dtype=np.float32)
        transform = np.eye(4)
        transform[:3, :] = t.reshape((3, 4))

        transform = y_down_to_y_up @ transform

        return transform

    @staticmethod
    def decode_intrinsic(session_configs: service_pb2.RegisterRequest):
        sx = session_configs.camera_color.resize_factor_x
        sy = session_configs.camera_color.resize_factor_y

        fx, fy = (
            session_configs.camera_intrinsics.focal_length_x * sx,
            session_configs.camera_intrinsics.focal_length_y * sy,
        )
        cx, cy = (
            session_configs.camera_intrinsics.principal_point_x * sx,
            session_configs.camera_intrinsics.principal_point_y * sy,
        )

        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return k

    @staticmethod
    def decode_point_cloud(
        session_configs: service_pb2.RegisterRequest,
        k: np.ndarray,
        color_rgb: np.ndarray,
        depth_img: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        # Flip image is needed for point cloud generation.
        color_rgb = np.flipud(color_rgb)
        depth_img = np.flipud(depth_img)

        color_img_w = int(session_configs.camera_intrinsics.resolution_x * session_configs.camera_color.resize_factor_x)
        color_img_h = int(session_configs.camera_intrinsics.resolution_y * session_configs.camera_color.resize_factor_y)
        u, v = np.meshgrid(np.arange(color_img_w), np.arange(color_img_h))
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]

        z = depth_img.copy()
        x = ((u - cx) * z) / fx
        y = ((v - cy) * z) / fy
        pcd = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        pcd = np.matmul(transform[:3, :3], pcd.T).T + transform[:3, 3]
        clr = color_rgb.reshape(-1, 3)

        return pcd, clr

    @staticmethod
    def deserialize_hand_tracking_data(buf: bytes) -> HandData:
        """
        Recreates the C# structure written in ARFlowMetaQuest.SerializeHandTrackingData.

        Returns
        -------
        HandData named-tuple with:
            left_tracked      : bool
            left_points       : list[(x,y,z) float]
            left_confidence   : int 0-255
            right_tracked     : bool
            right_points      : list[(x,y,z) float]
            right_confidence  : int 0-255
            timestamp         : float  (seconds since Unity started)
        """
        off = 0

        # ---- Header ------------------------------------------------------------
        if buf[0:4] != b"HAND":
            raise ValueError("Buffer does not start with 'HAND' header")
        off += 4

        # ---- Left hand ---------------------------------------------------------
        left_tracked = bool(buf[off])
        off += 1
        left_points, left_confidence = [], 0

        if left_tracked:
            (left_count,) = struct.unpack_from("<I", buf, off)  # uint32
            off += 4
            for _ in range(left_count):
                x, y, z = struct.unpack_from("<fff", buf, off)  # 3 × float32
                off += 12
                left_points.append((x, y, z))
            left_confidence = buf[off]  # 1-byte enum
            off += 4  # skip confidence + 3-byte padding

        # ---- Right hand --------------------------------------------------------
        right_tracked = bool(buf[off])
        off += 1
        right_points, right_confidence = [], 0

        if right_tracked:
            (right_count,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(right_count):
                x, y, z = struct.unpack_from("<fff", buf, off)
                off += 12
                right_points.append((x, y, z))
            right_confidence = buf[off]
            off += 4  # confidence + padding

        # ---- Timestamp ---------------------------------------------------------
        (timestamp,) = struct.unpack_from("<f", buf, off)  # float32

        return HandData(
            left_tracked, left_points, left_confidence, right_tracked, right_points, right_confidence, timestamp
        )
