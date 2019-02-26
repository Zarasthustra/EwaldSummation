# Ewald-Summation
Final project for the Computational Sciences Seminar

## Units and physical constants

We use the *real* unit system in [LAMMPS](https://lammps.sandia.gov/doc/units.html).

```
    mass = grams/mole
    distance = Angstroms
    time = femtoseconds
    energy = Kcal/mole
    velocity = Angstroms/femtosecond
    force = Kcal/mole-Angstrom
    torque = Kcal/mole
    temperature = Kelvin
    pressure = atmospheres
    dynamic viscosity = Poise
    charge = multiple of electron charge (1.0 is a proton)
    dipole = charge*Angstroms
    electric field = volts/Angstrom
    density = gram/cm^dim
```

So the involved physical constants are:
<a href="https://www.codecogs.com/eqnedit.php?latex=k_B=1.38064852\&space;J\cdot&space;K^{-1}\rightarrow&space;1.98720360\times10^{-3}\&space;kcal\cdot&space;mol^{-1}\cdot&space;K^{-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_B=1.38064852\&space;J\cdot&space;K^{-1}\rightarrow&space;1.98720360\times10^{-3}\&space;kcal\cdot&space;mol^{-1}\cdot&space;K^{-1}" title="k_B=1.38064852\ J\cdot K^{-1}\rightarrow 1.98720360\times10^{-3}\ kcal\cdot mol^{-1}\cdot K^{-1}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=k_C=\frac1{4\pi\epsilon}=8.987551787\times10^9\&space;J\cdot&space;m\cdot&space;C^{-2}\rightarrow&space;332.0637128\&space;kcal\cdot&space;\textrm{{\normalfont\AA}}\cdot&space;e^{-2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_C=\frac1{4\pi\epsilon}=8.987551787\times10^9\&space;J\cdot&space;m\cdot&space;C^{-2}\rightarrow&space;332.0637128\&space;kcal\cdot&space;\textrm{{\normalfont\AA}}\cdot&space;e^{-2}" title="k_C=\frac1{4\pi\epsilon}=8.987551787\times10^9\ J\cdot m\cdot C^{-2}\rightarrow 332.0637128\ kcal\cdot \textrm{{\normalfont\AA}}\cdot e^{-2}" /></a>


## How to run benchmarks / examples

Set root level of this repository as terminal working directory, and then run:

```
python3 -m ewald_summation.benchmark.bm_MD
python3 -m ewald_summation.benchmark.bm_Harmonic
```

