import matplotlib.pyplot as plt
import pandas as pd

gmres = pd.read_csv("./gmres_res.csv")
pbcgs = pd.read_csv("./pbicgstab_res.csv")

plt.plot(gmres["it"], gmres["res"])
plt.plot(pbcgs["it"], pbcgs["res"])

plt.show()