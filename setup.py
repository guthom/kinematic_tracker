import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='kinematic_tracker',
     version='0.1',
     scripts=[] ,
     author="Thomas Gulde",
     author_email="thomas.gulde@reutlingen-university.de",
     description="The kinematic tracker package for RoPose",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/guthom/kinematic_tracker",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
     ],
 )
