from distutils.core import setup
from Cython.Build import cythonize

setup(
    # ext_modules = cythonize(["Propagate.pyx", , "Field_Free.pyx", "Interaction.pyx", "Coefficent_Calculator.pyx"])
    ext_modules = cythonize(["Coefficent_Calculator.pyx", "Dipole_Acceleration_Matrix.pyx", "Interaction.pyx", "Field_Free_Matrix.pyx", "Propagate.pyx"])
)