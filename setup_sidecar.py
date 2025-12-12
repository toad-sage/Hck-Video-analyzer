from setuptools import setup, find_packages

setup(
    name="video-processor-sidecar",
    version="0.1",
    packages=['.'],
    py_modules=['sidecar_udf', 'kv_writer'],
    install_requires=[
        'couchbase>=4.0.0'
    ],
)

