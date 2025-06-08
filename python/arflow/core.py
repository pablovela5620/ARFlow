"""Data exchanging service."""

import os
import pickle
import re
import struct
import time
import uuid
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
        print("Processing frame")

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
            hands_data = self.parse_hand_data(request.depth)
            self._print_hand_poses(hands_data)


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
    def parse_hand_data(depth_bytes: bytes) -> dict | None:
        """Parse hand data from the depth bytes according to the specified format."""
        print(f"[DEBUG] parse_hand_data: Received {len(depth_bytes)} bytes")
        
        if len(depth_bytes) < 13:  # Minimum: header(4) + left_tracked(1) + right_tracked(1) + timestamp(4) + padding
            print(f"[DEBUG] parse_hand_data: Insufficient bytes ({len(depth_bytes)} < 13)")
            return None
            
        offset = 0
        
        # Bytes 0-3: "HAND" header
        header = depth_bytes[offset:offset+4].decode('ascii', errors='ignore')
        print(f"[DEBUG] parse_hand_data: Header = '{header}'")
        if header != "HAND":
            print(f"[DEBUG] parse_hand_data: Invalid header: {header}")
            return None
        offset += 4
        
        # Byte 4: Left hand tracked (0/1)
        left_tracked = struct.unpack('<B', depth_bytes[offset:offset+1])[0] == 1
        print(f"[DEBUG] parse_hand_data: Left hand tracked = {left_tracked}")
        offset += 1
        
        # Left hand data (if tracked)
        left_data = None
        if left_tracked:
            print(f"[DEBUG] parse_hand_data: Parsing left hand data at offset {offset}")
            if len(depth_bytes) < offset + 68:  # 60 bytes positions + 8 bytes state
                print(f"[DEBUG] parse_hand_data: Insufficient bytes for left hand data ({len(depth_bytes)} < {offset + 68})")
                return None
                
            # 5 finger positions: 5 fingers * 3 coordinates * 4 bytes = 60 bytes
            finger_positions = struct.unpack('<15f', depth_bytes[offset:offset+60])
            print(f"[DEBUG] parse_hand_data: Left hand finger positions = {finger_positions[:6]}... (showing first 6)")
            offset += 60
            
            # Parse left hand state: pinch_strength(4) + pinching(1) + grabbing(1) + confidence(1) + padding(1)
            pinch_strength = struct.unpack('<f', depth_bytes[offset:offset+4])[0]
            offset += 4
            pinching = struct.unpack('<B', depth_bytes[offset:offset+1])[0] == 1
            offset += 1
            grabbing = struct.unpack('<B', depth_bytes[offset:offset+1])[0] == 1
            offset += 1
            confidence = struct.unpack('<B', depth_bytes[offset:offset+1])[0]
            offset += 1
            offset += 1  # skip padding
            
            print(f"[DEBUG] parse_hand_data: Left hand state - pinch_strength={pinch_strength}, pinching={pinching}, grabbing={grabbing}, confidence={confidence}")
            
            # Organize finger positions as nested structure
            fingers = {
                'thumb': (finger_positions[0], finger_positions[1], finger_positions[2]),
                'index': (finger_positions[3], finger_positions[4], finger_positions[5]),
                'middle': (finger_positions[6], finger_positions[7], finger_positions[8]),
                'ring': (finger_positions[9], finger_positions[10], finger_positions[11]),
                'pinky': (finger_positions[12], finger_positions[13], finger_positions[14])
            }
            
            left_data = {
                'tracked': True,
                'fingers': fingers,
                'pinch_strength': pinch_strength,
                'pinching': pinching,
                'grabbing': grabbing,
                'confidence': confidence
            }
        else:
            left_data = {'tracked': False}
        
        # Right hand tracked (0/1)
        print(f"[DEBUG] parse_hand_data: Checking right hand at offset {offset}")
        if len(depth_bytes) < offset + 1:
            print(f"[DEBUG] parse_hand_data: Insufficient bytes for right hand tracking flag")
            return None
        right_tracked = struct.unpack('<B', depth_bytes[offset:offset+1])[0] == 1
        print(f"[DEBUG] parse_hand_data: Right hand tracked = {right_tracked}")
        offset += 1
        
        # Right hand data (if tracked)
        right_data = None
        if right_tracked:
            print(f"[DEBUG] parse_hand_data: Parsing right hand data at offset {offset}")
            if len(depth_bytes) < offset + 68:  # 60 bytes positions + 8 bytes state
                print(f"[DEBUG] parse_hand_data: Insufficient bytes for right hand data ({len(depth_bytes)} < {offset + 68})")
                return None
                
            # 5 finger positions: 5 fingers * 3 coordinates * 4 bytes = 60 bytes
            finger_positions = struct.unpack('<15f', depth_bytes[offset:offset+60])
            print(f"[DEBUG] parse_hand_data: Right hand finger positions = {finger_positions[:6]}... (showing first 6)")
            offset += 60
            
            # Parse right hand state: pinch_strength(4) + pinching(1) + grabbing(1) + confidence(1) + padding(1)
            pinch_strength = struct.unpack('<f', depth_bytes[offset:offset+4])[0]
            offset += 4
            pinching = struct.unpack('<B', depth_bytes[offset:offset+1])[0] == 1
            offset += 1
            grabbing = struct.unpack('<B', depth_bytes[offset:offset+1])[0] == 1
            offset += 1
            confidence = struct.unpack('<B', depth_bytes[offset:offset+1])[0]
            offset += 1
            offset += 1  # skip padding
            
            print(f"[DEBUG] parse_hand_data: Right hand state - pinch_strength={pinch_strength}, pinching={pinching}, grabbing={grabbing}, confidence={confidence}")
            
            # Organize finger positions as nested structure
            fingers = {
                'thumb': (finger_positions[0], finger_positions[1], finger_positions[2]),
                'index': (finger_positions[3], finger_positions[4], finger_positions[5]),
                'middle': (finger_positions[6], finger_positions[7], finger_positions[8]),
                'ring': (finger_positions[9], finger_positions[10], finger_positions[11]),
                'pinky': (finger_positions[12], finger_positions[13], finger_positions[14])
            }
            
            right_data = {
                'tracked': True,
                'fingers': fingers,
                'pinch_strength': pinch_strength,
                'pinching': pinching,
                'grabbing': grabbing,
                'confidence': confidence
            }
        else:
            right_data = {'tracked': False}
        
        # Timestamp (4 bytes - float)
        print(f"[DEBUG] parse_hand_data: Parsing timestamp at offset {offset}")
        if len(depth_bytes) < offset + 4:
            print(f"[DEBUG] parse_hand_data: Insufficient bytes for timestamp")
            return None
        timestamp = struct.unpack('<f', depth_bytes[offset:offset+4])[0]
        print(f"[DEBUG] parse_hand_data: Timestamp = {timestamp}")
        
        result = {
            'left_hand': left_data,
            'right_hand': right_data,
            'timestamp': timestamp
        }
        
        print(f"[DEBUG] parse_hand_data: Successfully parsed hand data - Left tracked: {left_data['tracked']}, Right tracked: {right_data['tracked']}")
        return result
    
    def _print_hand_poses(self, hand_data: dict):
        """Print hand pose information."""
        print(f"=== Hand Poses (Timestamp: {hand_data['timestamp']:.3f}) ===")
        
        # Left hand
        left = hand_data['left_hand']
        print(f"Left Hand: {'Tracked' if left['tracked'] else 'Not Tracked'}")
        if left['tracked']:
            print(f"  Positions: {left['fingers'][:5]}... (showing first 5)")
            print(f"  State: {left['pinch_strength']:.2f}, {left['pinching']}, {left['grabbing']}, {left['confidence']}")
            
        # Right hand  
        right = hand_data['right_hand']
        print(f"Right Hand: {'Tracked' if right['tracked'] else 'Not Tracked'}")
        if right['tracked']:
            print(f"  Positions: {right['fingers'][:5]}... (showing first 5)")
            print(f"  State: {right['pinch_strength']:.2f}, {right['pinching']}, {right['grabbing']}, {right['confidence']}")
        print()