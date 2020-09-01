from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["Potential.pyx", "Wigner.pyx", "Coefficent_Calculator.pyx", "Dipole_Acceleration_Matrix.pyx", "Interaction.pyx", "Field_Free_Matrix.pyx", "Propagate.pyx"])
)