from setuptools import setup

setup(
    name='WebvidReader',
    version='0.1.2',
    description='A package for reading the Webvid Dataset.',
    url='https://github.com/sesch023/WebvidReader',
    author='Sebastian Schmidt',
    author_email='schmidt.sebastian2@fh-swf.de',
    license='BSD 2-clause',
    packages=['WebvidReader'],
    install_requires=['pandas',
                      'opencv-python',
                      'torch'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
