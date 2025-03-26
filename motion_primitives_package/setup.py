from setuptools import find_packages, setup

package_name = 'motion_primitives_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jonathan',
    maintainer_email='j.rimini@icloud.com',
    description='Package related to Motion Planning using Motion Primitives',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'MotionPrimitivesNode = motion_primitives_package.MotionPrimitivesNode:main',
            'service = motion_primitives_package.service_member_function:main',
            'client = motion_primitives_package.client_member_function:main',
        ],
    },
)
