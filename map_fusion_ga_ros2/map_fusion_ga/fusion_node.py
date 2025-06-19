import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np

from .genetic_fusion import GeneticMapFusion


class MapFusionNode(Node):
    def __init__(self):
        super().__init__('map_fusion_node')
        self.sub1 = self.create_subscription(OccupancyGrid, '/map1', self.map1_callback, 10)
        self.sub2 = self.create_subscription(OccupancyGrid, '/map2', self.map2_callback, 10)
        self.publisher = self.create_publisher(OccupancyGrid, '/fused_map', 10)

        self.map1 = None
        self.map2 = None

    def map1_callback(self, msg):
        self.map1 = self.occupancygrid_to_numpy(msg)
        self.try_fusion(msg.header)

    def map2_callback(self, msg):
        self.map2 = self.occupancygrid_to_numpy(msg)
        self.try_fusion(msg.header)

    def try_fusion(self, header: Header):
        if self.map1 is not None and self.map2 is not None:
            fusion = GeneticMapFusion(self.map1, self.map2)
            best_params = fusion.evolve()
            fused_np, _, _ = fusion.fuse_maps_aligned(best_params)
            fused_ros = self.numpy_to_occupancygrid(fused_np, header)
            self.publisher.publish(fused_ros)
            self.map1 = None
            self.map2 = None  # Clear after publishing

    def occupancygrid_to_numpy(self, msg: OccupancyGrid) -> np.ndarray:
        data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        return (data > 50).astype(np.uint8)  # Binarize

    def numpy_to_occupancygrid(self, arr: np.ndarray, header: Header) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = 1.0
        msg.info.width = arr.shape[1]
        msg.info.height = arr.shape[0]
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = list((arr * 100).astype(np.int8).flatten())
        return msg
