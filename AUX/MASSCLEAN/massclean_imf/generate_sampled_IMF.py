
import numpy as np
from astropy.io import ascii
import subprocess

"""
Use MASSCLEAN's 'goimf2' to generate a file with mass sampled in small steps
until a total mass is generated
"""

mass_tot, mass_step = 10000, 10000

# Define IMF parameters in imf.ini file.
# IMPORTANT: (5) is the minimum mass sampled from the IMF. By default this is
# set to 0.15 because: "this is the lower limit of Padova models" (Bogdan,
# private email). This must be lowered to 0.01 (the lower limit of the IMF)
# otherwise mass is lost.
# "about half of the mass lies in the 0.01-0.15 M_Sun range", Bogdan
ini_file = """0.01       (1)
0.08       (2)
0.50       (3)
500.0      (4)
0.01       (5)
150        (6)
-99        (7)
0.3        (8)
1.3        (9)
2.35      (10)
1         (11)
"""
with open("ini.files/imf.ini", "w") as f:
    f.write(ini_file)

# Define total mass in cluster.ini file
ini_file = """0       (1)
{}    (2)
10    (3)
10    (4)
2048    (5)
2048    (6)
2048   (7)
2048   (8)
1       (9)
3.1    (10)
0.0    (11)
0.0    (12)
0.    (13)
0.0    (14)
0.0    (15)
0.0    (16)
0.0    (17)
0.0    (18)
0      (19)
0      (20)
0      (21)
0      (22)
0      (23)
""".format(mass_step)
with open("ini.files/cluster.ini", "w") as f:
    f.write(ini_file)

for _ in range(10):
    for mass in range(int(mass_tot / mass_step)):
        print(_, mass)

        # Call 'goimf2'
        bashCommand = "./goimf2"
        subprocess.call(bashCommand, stdout=subprocess.PIPE)

        # Read output
        sampled_IMF = ascii.read("n_distribution.out")['col1']

        with open("{}_IMF.dat".format(_), "a") as f:
            np.savetxt(f, sampled_IMF, fmt='%.5f')        
