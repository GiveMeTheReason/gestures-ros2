from setuptools import setup

package_name = 'mediapipe-extractor'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Timur Bayburin',
    maintainer_email='Timur.Bayburin@skoltech.ru',
    description='Mediapipe pose extractor from Azure Kinect',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mediapipe-extractor = scripts.pose_extractor:main',
        ],
    },
)
