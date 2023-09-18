# **A global apparent polar wander path for the last 320 Ma calculated from site-level paleomagnetic data**

### **by B. Vaes, D.J.J. van Hinsbergen, S.H.A. van de Lagemaat, E. van der Wiel, N. Lom, E.L. Advokaat, L.M. Boschman, L. C. Gall, A. Greve, C. Guilmette, S. Li, P.C. Lippert, L. Montheil, A. Qayyum & C.G. Langreis**

This manuscript was published in *Earth-Science Reviews* in 2023:

Vaes, B., van Hinsbergen, D. J., van de Lagemaat, S. H., van der Wiel, E., Lom, N., Advokaat, E. L., ... & Langereis, C. G. (2023). A global apparent polar wander path for the last 320 Ma calculated from site-level paleomagnetic data. Earth-Science Reviews, 104547. https://doi.org/10.1016/j.earscirev.2023.104547

-------
Repo for the data files and Python codes used to construct the global apparent polar wander path for the last 320 Ma from simulated site-level paleomagnetic data.

Tables S1, S2 and S3 are the supplementary data tables that accompany the paper. Table S1 contains the global plate circuit that is used to rotate the paleomagnetic data to a single reference plate. Table S2 contains the complete paleopole database used to compute the global APWP, with a range of statistical parameters and descriptions of the reliability and age determinations of individual paleopoles. Finally, Table S3 provides the global APWP computed at a 5 Ma resolution.

The Jupyter Notebook named Global_APWP.ipynb was used to perform to calculations and generate the figures of the paper.
To use the notebook, the following accompanying files are needed:
- Table_S2.xlsx: global paleomagnetic database, see description above
- APWP_functions.py: Python code with functions needed for the computation of the APWP
- Euler_poles_plate_circuit.csv: csv-file with Euler rotation poles computed per million year for each tectonic plate from which data is derived
- T12_gapwap.xlsx: global APWP by Torsvik et al. (2012, Earth-Science Reviews) - used as reference
- V23_gapwap.xlsx: global APWP presented in this study - used as reference


