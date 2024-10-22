import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from Tissue_compartment_modeling import CM_vB_wrap

if "Users" in os.getcwd():
    # Local runtime with data from Example_data folder
    print("Local runtime")
    WDIR = "./Example_data/"

elif "content" in os.getcwd():
    # Assume Colabs runtime
    # From: https://realpython.com/generative-adversarial-networks/
    # Mount google drive and set working directory (wdir)
    print("Colabs runtime")
    from google.colab import drive

    drive.mount("/content/drive/")
    WDIR = "/content/drive/My Drive/Jobb/PET_ML_Research/Kurs & Konferens/Physics in Nuclear Medicine Special Curriculum/"

PATH_AIF = WDIR + "AIF_shift_SUV_M3.txt"
PATH_VOI = WDIR + "VOIdata_M3.voistat"

# %% Load AIF
with open(PATH_AIF, "rb") as f:
    df = pd.read_csv(f, sep="\t")

COLS = df.columns

AIF_t = np.array(df.loc[:, [COLS[0]]]).squeeze()
AIF_t = AIF_t - AIF_t[0]  # Zero shift
AIF_A = np.array(df.loc[:, [COLS[1]]]).squeeze()

# %% Simulate a tissue curve
# One tissue compartment model
num_k = 4
# CM_vB = CM_vB_wrap(num_k
CM_vB_1TCM = CM_vB_wrap(2)
CM_vB_2TCM_irrev = CM_vB_wrap(3)
CM_vB_2TCM_rev = CM_vB_wrap(4)
K1 = 0.1
k2 = 0.05
k3 = 0.1
k4 = 0.05
vB = 0

Ct_1TCM = CM_vB_1TCM((AIF_A, AIF_t), K1, k2, vB)
Ct_2TCM_irrev = CM_vB_2TCM_irrev((AIF_A, AIF_t), K1, k2, vB, k3)
Ct_2TCM_rev = CM_vB_2TCM_rev((AIF_A, AIF_t), K1, k2, vB, k3, k4)
# if num_k == 2:
#     tcm = "1TCM"
#     Ct = CM_vB((AIF_A, AIF_t), K1, k2, vB)
#     plt_title = tcm + f", K1={K1}, k2={k2}, vB={vB}"
# elif num_k == 3:
#     tcm = "2TCM_irrev"
#     Ct = CM_vB((AIF_A, AIF_t), K1, k2, vB, k3)
#     plt_title = tcm + f", K1={K1}, k2={k2}, k3={k3}, vB={vB}"
# elif num_k == 4:
#     tcm = "2TCM_rev"
#     Ct = CM_vB((AIF_A, AIF_t), K1, k2, vB, k3, k4)
#     plt_title = tcm + f", K1={K1}, k2={k2}, k3={k3}, k4={k4}, vB={vB}"

plt.plot(AIF_t, AIF_A, label="AIF")
plt.plot(AIF_t, Ct_1TCM, label="1TCM")
plt.plot(AIF_t, Ct_2TCM_irrev, label="2TCM_irrev")
plt.plot(AIF_t, Ct_2TCM_rev, label="2TCM_rev")
# plt.plot(AIF_t, Ct, label="Tissue curve")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "K1="
    + str(K1)
    + ", k2="
    + str(k2)
    + ", k3="
    + str(k3)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)
# %% Varying K1
K1_range = np.linspace(0.01, 0.5, 5)
k2 = 0.05
k3 = 0.1
k4 = 0.05
vB = 0
Ct_1TCM = np.zeros((len(K1_range), len(AIF_t)))
Ct_2TCM_irrev = np.zeros((len(K1_range), len(AIF_t)))
Ct_2TCM_rev = np.zeros((len(K1_range), len(AIF_t)))
for idx, K1 in enumerate(K1_range):
    Ct_1TCM[idx] = CM_vB_1TCM((AIF_A, AIF_t), K1, k2, vB)
    Ct_2TCM_irrev[idx] = CM_vB_2TCM_irrev((AIF_A, AIF_t), K1, k2, vB, k3)
    Ct_2TCM_rev[idx] = CM_vB_2TCM_rev((AIF_A, AIF_t), K1, k2, vB, k3, k4)

# Plot 1TCM
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, K1 in enumerate(K1_range):
    plt.plot(AIF_t, Ct_1TCM[idx], label=f"K1={K1}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "1TCM, "
    + "k2="
    + str(k2)
    + ", k3="
    + str(k3)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)

# Plot 2TCM_irrev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, K1 in enumerate(K1_range):
    plt.plot(AIF_t, Ct_2TCM_irrev[idx], label=f"K1={K1}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_irrev, "
    + "k2="
    + str(k2)
    + ", k3="
    + str(k3)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)

# Plot 2TCM_rev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, K1 in enumerate(K1_range):
    plt.plot(AIF_t, Ct_2TCM_rev[idx], label=f"K1={K1}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_rev, "
    + "k2="
    + str(k2)
    + ", k3="
    + str(k3)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)

# %%
# Varying k2
K1 = 0.1
k2_range = np.linspace(0.01, 0.5, 5)
k3 = 0.1
k4 = 0.05
vB = 0
Ct_1TCM = np.zeros((len(k2_range), len(AIF_t)))
Ct_2TCM_irrev = np.zeros((len(k2_range), len(AIF_t)))
Ct_2TCM_rev = np.zeros((len(k2_range), len(AIF_t)))
for idx, k2 in enumerate(k2_range):
    Ct_1TCM[idx] = CM_vB_1TCM((AIF_A, AIF_t), K1, k2, vB)
    Ct_2TCM_irrev[idx] = CM_vB_2TCM_irrev((AIF_A, AIF_t), K1, k2, vB, k3)
    Ct_2TCM_rev[idx] = CM_vB_2TCM_rev((AIF_A, AIF_t), K1, k2, vB, k3, k4)

# Plot 1TCM
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k2 in enumerate(k2_range):
    plt.plot(AIF_t, Ct_1TCM[idx], label=f"k2={k2}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "1TCM, "
    + "K1="
    + str(K1)
    + ", k3="
    + str(k3)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)
# Plot 2TCM_irrev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k2 in enumerate(k2_range):
    plt.plot(AIF_t, Ct_2TCM_irrev[idx], label=f"k2={k2}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_irrev, "
    + "K1="
    + str(K1)
    + ", k3="
    + str(k3)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)
# Plot 2TCM_rev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k2 in enumerate(k2_range):
    plt.plot(AIF_t, Ct_2TCM_rev[idx], label=f"k2={k2}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_rev, "
    + "K1="
    + str(K1)
    + ", k3="
    + str(k3)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)
# %%
# Varying k3
K1 = 0.1
k2 = 0.05
k3_range = np.linspace(0.01, 0.5, 5)
k4 = 0.05
vB = 0
Ct_1TCM = np.zeros((len(k3_range), len(AIF_t)))
Ct_2TCM_irrev = np.zeros((len(k3_range), len(AIF_t)))
Ct_2TCM_rev = np.zeros((len(k3_range), len(AIF_t)))
for idx, k3 in enumerate(k3_range):
    Ct_1TCM[idx] = CM_vB_1TCM((AIF_A, AIF_t), K1, k2, vB)
    Ct_2TCM_irrev[idx] = CM_vB_2TCM_irrev((AIF_A, AIF_t), K1, k2, vB, k3)
    Ct_2TCM_rev[idx] = CM_vB_2TCM_rev((AIF_A, AIF_t), K1, k2, vB, k3, k4)

# Plot 1TCM
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k3 in enumerate(k3_range):
    plt.plot(AIF_t, Ct_1TCM[idx], label=f"k3={k3}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "1TCM, "
    + "K1="
    + str(K1)
    + ", k2="
    + str(k2)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)
# Plot 2TCM_irrev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k3 in enumerate(k3_range):
    plt.plot(AIF_t, Ct_2TCM_irrev[idx], label=f"k3={k3}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_irrev, "
    + "K1="
    + str(K1)
    + ", k2="
    + str(k2)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)
# Plot 2TCM_rev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k3 in enumerate(k3_range):
    plt.plot(AIF_t, Ct_2TCM_rev[idx], label=f"k3={k3}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_rev, "
    + "K1="
    + str(K1)
    + ", k2="
    + str(k2)
    + ", k4="
    + str(k4)
    + ", vB="
    + str(vB)
)
# %%
# Varying k4
K1 = 0.1
k2 = 0.05
k3 = 0.1
k4_range = np.linspace(0.01, 0.5, 5)
vB = 0
Ct_1TCM = np.zeros((len(k4_range), len(AIF_t)))
Ct_2TCM_irrev = np.zeros((len(k4_range), len(AIF_t)))
Ct_2TCM_rev = np.zeros((len(k4_range), len(AIF_t)))
for idx, k4 in enumerate(k4_range):
    Ct_1TCM[idx] = CM_vB_1TCM((AIF_A, AIF_t), K1, k2, vB)
    Ct_2TCM_irrev[idx] = CM_vB_2TCM_irrev((AIF_A, AIF_t), K1, k2, vB, k3)
    Ct_2TCM_rev[idx] = CM_vB_2TCM_rev((AIF_A, AIF_t), K1, k2, vB, k3, k4)

# Plot 1TCM
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k4 in enumerate(k4_range):
    plt.plot(AIF_t, Ct_1TCM[idx], label=f"k4={k4}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "1TCM, "
    + "K1="
    + str(K1)
    + ", k2="
    + str(k2)
    + ", k3="
    + str(k3)
    + ", vB="
    + str(vB)
)
# Plot 2TCM_irrev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k4 in enumerate(k4_range):
    plt.plot(AIF_t, Ct_2TCM_irrev[idx], label=f"k4={k4}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_irrev, "
    + "K1="
    + str(K1)
    + ", k2="
    + str(k2)
    + ", k3="
    + str(k3)
    + ", vB="
    + str(vB)
)
# Plot 2TCM_rev
plt.figure()
plt.plot(AIF_t, AIF_A, label="AIF")
for idx, k4 in enumerate(k4_range):
    plt.plot(AIF_t, Ct_2TCM_rev[idx], label=f"k4={k4}")
plt.xlabel("Time [min]")
plt.ylabel("SUV (g/mL)")
plt.legend()
plt.title(
    "2TCM_rev, "
    + "K1="
    + str(K1)
    + ", k2="
    + str(k2)
    + ", k3="
    + str(k3)
    + ", vB="
    + str(vB)
)
# %%
