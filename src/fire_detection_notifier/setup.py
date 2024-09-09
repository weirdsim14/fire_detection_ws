from setuptools import setup

package_name = 'fire_detection_notifier'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS2 package for fire detection with publishing',
    license='Apache License 2.0', tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fire_detection = fire_detection_notifier.detect:main',  # Update this line to point to your script
        ],
    },
)
