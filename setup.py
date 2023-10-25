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
    install_requires=['pandas',
                      'decord',
                      'torch',
                      'opencv-python',
                      'einops'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7+',
    ],
)
