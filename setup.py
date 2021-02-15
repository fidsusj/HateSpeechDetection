""" setup.py """

from setuptools import find_packages, setup

setup(
    name="HateSpeechDetectionSrc",
    version="1.0.0",
    description="Hate speech detection project; course Text Analytics at Uni Heidelberg",
    author="Felix Hausberger, Christopher Klammt, Nils Krehl",
    url="https://github.com/fidsusj/HateSpeechDetection",
    packages=find_packages(),
    test_suite="nose.collector",
    tests_require=["nose"],
)
