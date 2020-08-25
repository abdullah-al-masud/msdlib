import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="msdlib",
    version="0.0.3.2",
    author="Abdullah Al Masud",
    author_email="abdullahalmasud.buet@gmail.com",
    description="My utility functions are stored here which I often use for my purposes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abdullah-al-masud/msdLib",
    packages=['msdlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
