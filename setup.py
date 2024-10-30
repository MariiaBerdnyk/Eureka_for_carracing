from setuptools import setup, find_packages

# Load dependencies from requirements.txt
with open("requirements.txt") as f:
    INSTALL_REQUIRES = f.read().splitlines()

# Installation operation
setup(
    name="eureka",
    version="1.0",
    description="Eureka",
    keywords=["llm", "rl"],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
)
