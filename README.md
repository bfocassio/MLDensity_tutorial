Full files with CHG densities: https://www.dropbox.com/s/hhv2klie7slp833/tutorial_MLChargeDensity.zip?dl=0

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

