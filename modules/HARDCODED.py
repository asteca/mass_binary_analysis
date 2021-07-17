
#
# Only Gaia EDR3 and Gaia DR2 photometric systems are supported
cmd_systs = {
    'gaiadr2': (
        ('Gmag', 'G_BPmag', 'G_RPmag'), (6437.7, 5309.57, 7709.85)),
    'gaiaedr3': (
        ('Gmag', 'G_BPmag', 'G_RPmag'), (6422.01, 5335.42, 7739.17))
}
# line where the header starts in the CMD isochrone files
idx_header = 11
# Column names in isochrone files
logAge_name, label_name = 'logAge', 'label'
# Column names
mag_name, col_name = 'Gmag', 'BP-RP'
