from setuptools import setup

package_name = 'map_fusion_ga'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/fusion.launch.py']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'numpy',
        'scipy',
        'scikit-image',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='C. Luna',
    maintainer_email='cristina.luna@live.com',
    description='ROS 2 node for occupancy grid map fusion using a genetic algorithm',
    license='GNU GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_node = map_fusion_ga.fusion_node:main',
        ],
    },
)
