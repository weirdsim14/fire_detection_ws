import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class FireDetectionPublisher(Node):
    def __init__(self):
        super().__init__('fire_detection_publisher')
        self.publisher_ = self.create_publisher(Bool, 'fire_detected', 10)

    def publish_fire_detection(self, detected):
        msg = Bool()
        msg.data = detected
        self.publisher_.publish(msg)
