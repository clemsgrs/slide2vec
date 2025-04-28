# Steps to set up testing environment
Set up conda environment:

```
conda create --name=slide2vec python=3.12
```

Activate environment:

```
conda activate slide2vec
```

Install module and additional development dependencies:

```
pip install -e .
pip install -r requirements_dev.txt
```

Perform tests:

```
./do_build.sh
```

# Programming conventions
AutoPEP8 for formatting (this can be done automatically on save, see e.g. https://code.visualstudio.com/docs/python/editing)

# Push release to PyPI
1. Increase version in setup.py, and set below
2. Build: `python -m build`
3. Distribute package to PyPI: `python -m twine upload dist/*0.0.1*`
