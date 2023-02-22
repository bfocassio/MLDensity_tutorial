Paper available on arXiv: https://arxiv.org/abs/2301.13550

Full files with CHG densities: https://zenodo.org/record/7599897#.Y_ZqoNJKiV4

Slides: https://docs.google.com/presentation/d/19w3Ekzgh7BMP_uck4IEirDW5GcV74V8xy7g3HHsiVDk/edit?usp=sharing

First, install required packages:

```
conda env create --name jlchg_tutorial --file=jlgridfingerprints_environment.yml
conda activate jlchg_tutorial
```

Second, to compile cython files within the code use from the `jlgridfingerprints` directory:

```
LDSHARED="gcc -shared" CC=gcc python setup.py build_ext --inplace
```

Third, play with the examples

