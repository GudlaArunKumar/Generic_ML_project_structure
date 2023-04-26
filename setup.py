from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)-> List[str]:
    """
    Returns list of libraries / requirements to setup function
    """
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    
        # to remove the -e . in requirements list
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements


setup(
name='ML_generic_project',
version='0.0.1',
author='Arun Kumar Gudla',
author_email='arunkumar.gudla98@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)