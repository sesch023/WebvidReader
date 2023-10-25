from setuptools import setup

setup(
    name='WebvidReader',
    version='0.1.11',
    description='A package for reading the Webvid Dataset.',
    url='https://github.com/sesch023/WebvidReader',
    author='Sebastian Schmidt',
    author_email='schmidt.sebastian2@fh-swf.de',
    license='BSD 2-clause',
    packages=['WebvidReader'],
    install_requires=['pandas==1.5.3',
                      'decord==0.6.0',
                      'torch==2.0.0',
                      'opencv-python==4.8.0.74',
                      'einops==0.7.0'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7+',
    ],
)
