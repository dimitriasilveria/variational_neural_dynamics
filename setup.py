from setuptools import setup, find_packages

setup(
    name="lotf",
    version="0.1.0",
    description="A Python package to learn agile flight using differentiable simulation.",
    author="Michael Pan",
    author_email="michael.pan31415@gmail.com",
    packages=find_packages(),
    package_data={"lotf.envs": ["wind_z_table.csv"]},
    install_requires=[],
)
