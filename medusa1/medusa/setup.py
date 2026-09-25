from setuptools import find_packages, setup

setup(
    name="medusa",
    version="0.1.0",
    description="Re-implementation of MEDUSA (Cai et al., 2024): multiple decoding heads + tree attention",
    packages=find_packages(include=["medusa", "medusa.*"]),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "bitsandbytes==0.45.0",
        "peft==0.13.2",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "numpy<2",
        "pyyaml",
        "tqdm",
    ],
)
