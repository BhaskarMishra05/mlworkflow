import os
from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirement=[]
    with open (file_path) as file_obj:
        requirement= file_obj.readlines()
        requirement= [req.replace('\n',"") for req in requirement]
    return requirement

setup  (
    name="ml-project",
    version= '0.0.1',
    author='Bhaskar Mishra',
    author_email='bhaskarmishra1590@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirement.txt')
)