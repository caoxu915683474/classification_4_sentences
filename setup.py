from setuptools import setup, find_packages

setup(
    name="PineappleCake",
    version="0.1",
    author="ChuangXin-NLP",
    url="http://gitlab.ai.chuangxin.com/NLP/PineappleCake",
    packages=find_packages(),
    install_requires=[
        "tensorflow-gpu",
        "keras",
        "numpy",
        "pandas",
        "sklearn",
        "nltk",
        "tqdm",
        "gensim",
        "textrank4zh",
        "matplotlib",
        "six",
        "hanziconv"
    ]
    
)
