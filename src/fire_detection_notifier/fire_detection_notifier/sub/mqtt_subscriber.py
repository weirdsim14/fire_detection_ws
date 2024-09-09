# mqtt_subscriber.py
import json
import logging
import paho.mqtt.client as mqtt
from config import MQTTConfig
from threading import Thread
import paho.mqtt.client as mqttClient

class MQTT_SUB:
    def __init__(self, sound_player, qos=1):
        self.sound_player = sound_player
        self.client = mqtt.Client(mqttClient.CallbackAPIVersion.VERSION1)
        self.qos = qos
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(MQTTConfig.BROKER, MQTTConfig.PORT)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected successfully.")
            self.client.subscribe(MQTTConfig.TOPIC, self.qos)
        else:
            logging.error(f"Failed to connect, return code {rc}\n")
        
    def on_message(self, client, userdata, msg):
        logging.info("Received message on {msg.topic}: {msg.payload.decode('utf-8')}")
        payload_data = json.loads(msg.payload.decode('utf-8'))

        # Database insert operation

        # Sound playing logic
        if not self.sound_player.sound_lock.locked():
            Thread(target=self.sound_player.play_sound).start()
            
    def run(self):
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Script interrupted by user")
        finally:
            self.client.disconnect()