from setuptools import setup, find_packages

setup(
    name='live_tracking',
    description='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cv2',
        'torchvision',
        'gdown', # reid
        'loguru', # Byte
        'thop', # Byte
        'yolox', # Byte
        'lap', # Byte
        'cython_bbox', # Byte / for window: pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
    ],
)