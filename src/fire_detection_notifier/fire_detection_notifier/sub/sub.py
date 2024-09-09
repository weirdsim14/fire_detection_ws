import logging
from sound_player import SoundPlayer
from mqtt_subscriber import MQTT_SUB

if __name__ == '__main__':
    sound_player = SoundPlayer("warning.wav")
    mqtt_sub = MQTT_SUB(sound_player=sound_player)
    mqtt_sub.run()