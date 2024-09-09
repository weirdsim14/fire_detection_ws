import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from threading import Timer

class FireDetectionSubscriber(Node):
    def __init__(self):
        super().__init__('fire_detection_subscriber')

        self.subscription = self.create_subscription(
            Bool,
            'fire_detected',
            self.fire_detected_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        self.sound_command_publisher = self.create_publisher(
            String,
            '/command/play_sound',
            10
        )

        # State to track if sound is currently playing
        self.is_sound_playing = False
        self.sound_duration = 4.63  # Duration of the warning sound in seconds (adjust accordingly)
        self.stop_timer = None

    def fire_detected_callback(self, msg):
        if msg.data:
            if not self.is_sound_playing:  # Check if the sound is already playing
                self.get_logger().info('Fire detected! Publishing play sound command...')

                # Create the message to play the warning sound
                sound_command_msg = String()

                # Ensure the message is a valid string
                try:
                    sound_command_msg.data = ':/home/ghost/current_ros2/share/ghost_comms2/sounds/warning.wav'
                    assert isinstance(sound_command_msg.data, str), "Data is not a string!"
                    assert sound_command_msg.data, "String data is empty!"

                    # Publish the play sound command
                    self.sound_command_publisher.publish(sound_command_msg)
                    self.get_logger().info('Play sound command published.')

                    # Set state to indicate sound is playing
                    self.is_sound_playing = True

                    # Start a timer to reset the state after the sound duration
                    self.stop_timer = Timer(self.sound_duration, self.reset_sound_state)
                    self.stop_timer.start()

                except AssertionError as e:
                    self.get_logger().error(f'Error creating message: {e}')
                except Exception as e:
                    self.get_logger().error(f'Unexpected error: {e}')

    def reset_sound_state(self):
        """Resets the sound playing state after the duration of the sound."""
        self.is_sound_playing = False
        self.get_logger().info('Sound finished playing. Ready for next command.')

        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

def main(args=None):
    rclpy.init(args=args)

    fire_detection_subscriber = FireDetectionSubscriber()

    rclpy.spin(fire_detection_subscriber)

    fire_detection_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
