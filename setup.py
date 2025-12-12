from setuptools import setup

setup(
    name='video_processor_udf',
    version='0.2',
    py_modules=['main_udf', 'video_ai', 'kv_writer'],
    install_requires=[
        'couchbase',
        'ultralytics', 
        'opencv-python-headless',
        'transformers',
        'qwen-vl-utils',
        'sentence-transformers',
        'accelerate'
    ],
)
