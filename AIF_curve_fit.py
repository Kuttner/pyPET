import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import auc, r2_score
from scipy import stats
from scipy.odr import Model, Data, ODR


def AIF_fit(
    t,
    AIFm,
    RETURN_FIT_PARAMS=False,
    PLOT_DISPERSION=False,
    SAVE_FIG=False,
    SAVE_PATH=None,
    SAVE_CSV=False,
):
    # Function that fits the Feng1993 AIF function to measured AIF data
    # A fit is performed to the measured data AIFm, and returned as AIFm_fit,
    #
    # Dispersion correction is also performed using a single exponential dispersion model
    #
    # The dispersed curve (g(t)) is modeled as:
    # g(t) = AIFu x d(t)
    # where AIFu is the dispersion corrected AIFm and d(t) = (1/tau)*exp(-t/tau)
    # is the dispersion function and (x) stands for convolution.
    #
    # Input:    t           Time vector in minutes (If seconds are passed, its converted)
    #           AIFm        The measured AIF
    #           FLAGS       Plot and save flags
    #
    # Output    AIFm_fit    The Feng1993 fit to AIFm
    #           AIFu        The undispersed (corrected) AIF
    #                       (Neglect this output if an IDIF is fitted)
    #
    # Written by Samuel Kuttner 2022-12-01
    #

    # Check if time vector could be in seconds, then convert to minutes
    if np.max(t) > 300:
        t = t / 60

    # %% Fit the measured AIF: AIFm
    # params_init = [0, 141, 0.296, 0.259, -27, -0.55, -0.04]
    params_init = [0.975, 80.26, 1, 2.18, -5.235, -0.071, -0.189]
    llim = [0, 0, 0, 0, -100, -1, -1]
    ulim = [1.5, 5100, 999, 99, 9, 9, 0]

    AIFm_fit_params, pcov, infodict, mesg, ier = curve_fit(
        feng_aif,
        t,
        AIFm,
        method="trf",
        bounds=(llim, ulim),
        p0=params_init,
        maxfev=5000,
        full_output=True,
    )
    # print(AIFm_fit_params)
    TAUm, A1m, A2m, A3m, L1m, L2m, L3m = AIFm_fit_params
    AIFm_fit = feng_aif(t, TAUm, A1m, A2m, A3m, L1m, L2m, L3m)
    r2m = r_squared(AIFm, AIFm_fit)

    # %% Find the undispersed input function: AIFu
    # Use same limits and initial conditions, but add for TAUd.
    initTAUd = 0.12
    llimTAUd = 0
    ulimTAUd = 1

    llim.append(llimTAUd)
    ulim.append(ulimTAUd)
    params_init.append(initTAUd)

    AIFd_fit_params, pcov2, infodict2, mesg2, ier2 = curve_fit(
        AIFd_fcn,
        t,
        AIFm,
        method="trf",
        bounds=(llim, ulim),
        p0=params_init,
        maxfev=5000,
        full_output=True,
    )

    TAUu, A1u, A2u, A3u, L1u, L2u, L3u, TAUd = AIFd_fit_params

    AIFu = feng_aif(t, TAUu, A1u, A2u, A3u, L1u, L2u, L3u)
    AIFd = AIFd_fcn(t, TAUu, A1u, A2u, A3u, L1u, L2u, L3u, TAUd)
    r2u = r_squared(AIFm, AIFd)

    # %% Plot
    X_axis = (3000, 300)
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, t_max in enumerate(X_axis):
        if idx == 1:
            inset_axes(
                ax,
                width="100%",  # width = 30% of parent_bbox
                height="100%",  # height : 1 inch
                loc=1,
                bbox_to_anchor=(0.45, 0.5, 0.5, 0.5),
                bbox_transform=ax.transAxes,
            )
        plt.plot(t[:t_max], AIFm[:t_max], label="Measured AIF (AIFm)")
        plt.plot(t[:t_max], AIFm_fit[:t_max], label="AIFm_fit")
        if PLOT_DISPERSION:
            plt.plot(t[:t_max], AIFu[:t_max], label="Undispersed AIF (AIFu)")
            plt.plot(t[:t_max], AIFd[:t_max], label="Dispersed AIF (AIFu x D) ")
        if idx == 0:
            plt.title(SAVE_PATH)
        else:
            plt.legend()
        plt.xlabel("Time [min]")
        plt.ylabel("Activity")
    plt.show()
    if SAVE_FIG:
        print(SAVE_PATH)
        save_path = SAVE_PATH + "AIFs.pdf"
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    # %% Save csv file with corrected data
    if SAVE_CSV:
        save_path = SAVE_PATH + "_PY_disp_corr.csv"
        df2 = pd.DataFrame(
            {
                "Time [min]": t,
                "Measured AIF (AIFm)": AIFm,
                "AIFm_fit": AIFm_fit,
                "Undispersed AIF (AIFu)": AIFu,
                "Dispersed AIF (AIFu x D)": AIFd,
            }
        )
        df2.to_csv(save_path, index=False)

    if RETURN_FIT_PARAMS:
        return AIFm_fit, AIFm_fit_params
    else:
        return AIFm_fit, AIFu


# %%
def AIF_fit_weights_disp(
    t,
    AIFm,
    INTERPOLATE=False,
    TIME_FRAMES=1,
    PLOT_DISPERSION=False,
    SAVE_FIG=False,
    SAVE_PATH=None,
    SAVE_CSV=False,
    DISP_CORR_X_MIN=0,  # 0 minutes means all data is used for dispersion correction
):
    # Function that fits the Feng1993 AIF function to measured AIF data
    # A fit is performed to the measured data AIFm, and returned as AIFm_fit,
    #
    # This fit function uses a weighted objective function:
    #
    # L(AIFm_fit - AIFm*w)
    #
    # where:
    #
    # AIFm_fit is the fitted curve,
    # AIFm is the measured curve, and
    # w is the weight, w = AIFm^1/n,
    #
    # where
    #
    # n = 1:21
    #
    # This fit is very sensitive to starting parameters (params_init) and to
    # the lower and upper limits (llim, ulim). The suggested values work pretty well.
    #
    # Dispersion correction is also performed using a single exponential dispersion model
    #
    # The dispersed curve (g(t)) is modeled as:
    # g(t) = AIFu x d(t)
    # where AIFu is the dispersion corrected AIFm and d(t) = (1/tau)*exp(-t/tau)
    # is the dispersion function and (x) stands for convolution.
    #
    # Input:    t                   Time vector in minutes (If seconds are passed, its converted)
    #           AIFm                The measured AIF
    #           FLAGS               Plot and save flags
    #           DISP_CORR_X_MIN     If > 0, then only the first X minutes of AIFm are used for dispersion correction, but all values are saved
    #
    # Output    AIFm_fit            The Feng1993 fit to AIFm
    #           AIFu                The undispersed (corrected) AIF
    #                               (Neglect this output if an IDIF is fitted)
    #
    # Written by Samuel Kuttner 2022-12-01
    #

    # Check if time vector could be in seconds, then convert to minutes
    if np.max(t) > 60:
        t = t / 60

    # Set negative values to zero
    AIFm[AIFm < 0] = 0

    if INTERPOLATE:
        AIFm, _, t_int = interpolate_time_frames(
            AIFm, t * 60, AIFm, t * 60, TIME_FRAMES
        )
        t_org = t
        t = t_int / 60
    # %% Fit the measured AIF: AIFm
    # params_init = [0, 141, 0.296, 0.259, -27, -0.55, -0.04]
    params_init = [0.975, 80.26, 1, 2.18, -5.235, -0.071, -0.189]
    llim = [0, 0, 0, 0, -100, -1, -1]
    ulim = [1.5, 5100, 999, 99, 9, 9, 0]

    J_max = []  # List to store the max values
    AIFm_fit_params_list = []
    AIFm_fit_list = []

    for n in range(1, 100):
        weight = np.power(AIFm, 1 / n)

        AIFm_fit_params, pcov, infodict, mesg, ier = curve_fit(
            feng_aif,
            t,
            np.multiply(AIFm, weight),
            method="trf",
            bounds=(llim, ulim),
            p0=params_init,
            maxfev=2**14,
            full_output=True,
        )
        # print(AIFm_fit_params)
        AIFm_fit_params_list.append(AIFm_fit_params)
        TAUm, A1m, A2m, A3m, L1m, L2m, L3m = AIFm_fit_params
        AIFm_fit = feng_aif(t, TAUm, A1m, A2m, A3m, L1m, L2m, L3m)
        # r2m = r_squared(AIFm, AIFm_fit)
        AIFm_fit_list.append(AIFm_fit)
        J_max.append(((np.max(AIFm_fit) - np.max(AIFm)) ** 2))

    # Find optimal nthroot, as index of minimum J
    optimal_n = np.argmin(J_max)
    AIFm_fit_params_optimal = AIFm_fit_params_list[optimal_n]
    AIFm_fit_optimal = AIFm_fit_list[optimal_n]
    print("Optimal n: ", optimal_n)
    # %% Find the undispersed input function: AIFu
    # Use same limits and initial conditions, but add for TAUd.
    initTAUd = 0.12
    llimTAUd = 0
    ulimTAUd = 1

    llim.append(llimTAUd)
    ulim.append(ulimTAUd)
    params_init.append(initTAUd)

    J_max = []  # List to store the max values
    AIFd_fit_params_list = []
    AIFd_fit_list = []
    AIFu_list = []

    # If DISP_CORR_X_MIN is True, then only fit the first X minutes of AIFm
    if DISP_CORR_X_MIN > 0:
        t_org = t.copy()
        AIFm_org = AIFm.copy()
        t = t[t_org < DISP_CORR_X_MIN]
        AIFm = AIFm[t_org < DISP_CORR_X_MIN]
    else:
        t_org = t.copy()
        AIFm_org = AIFm.copy()

    for n in range(1, 60):
        weight = np.power(AIFm, 1 / n)

        AIFd_fit_params, pcov2, infodict2, mesg2, ier2 = curve_fit(
            AIFd_fcn,
            t,
            np.multiply(AIFm, weight),
            method="trf",
            bounds=(llim, ulim),
            p0=params_init,
            maxfev=2**14,
            full_output=True,
        )

        AIFd_fit_params_list.append(AIFd_fit_params)

        TAUu, A1u, A2u, A3u, L1u, L2u, L3u, TAUd = AIFd_fit_params

        AIFu = feng_aif(t_org, TAUu, A1u, A2u, A3u, L1u, L2u, L3u)
        AIFd_fit = AIFd_fcn(t_org, TAUu, A1u, A2u, A3u, L1u, L2u, L3u, TAUd)
        # r2u = r_squared(AIFm, AIFd)
        AIFd_fit_list.append(AIFd_fit)
        AIFu_list.append(AIFu)
        J_max.append(((np.max(AIFd_fit) - np.max(AIFm)) ** 2))

    # Find optimal nthroot, as index of minimum J
    optimal_n = np.argmin(J_max)
    AIFd_fit_params_optimal = AIFd_fit_params_list[optimal_n]
    (
        TAUuopt,
        A1dopt,
        A2dopt,
        A3dopt,
        L1dopt,
        L2dopt,
        L3dopt,
        TAUdopt,
    ) = AIFd_fit_params_optimal
    AIFu_optimal = AIFu_list[optimal_n]
    AIFd_optimal = AIFd_fcn(
        t_org, TAUuopt, A1dopt, A2dopt, A3dopt, L1dopt, L2dopt, L3dopt, TAUdopt
    )
    print("Optimal n: ", optimal_n)
    # %% Plot
    X_axis = (3000, 300)
    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, t_max in enumerate(X_axis):
        if idx == 1:
            inset_axes(
                ax,
                width="100%",  # width = 30% of parent_bbox
                height="100%",  # height : 1 inch
                loc=1,
                bbox_to_anchor=(0.45, 0.5, 0.5, 0.5),
                bbox_transform=ax.transAxes,
            )
        plt.plot(t[:t_max], AIFm[:t_max], label="Measured AIF (AIFm)")
        plt.plot(t_org[:t_max], AIFm_fit[:t_max], label="AIFm_fit")
        if PLOT_DISPERSION:
            plt.plot(
                t_org[:t_max], AIFu_optimal[:t_max], label="Undispersed AIF (AIFu)"
            )
            plt.plot(
                t_org[:t_max], AIFd_optimal[:t_max], label="Dispersed AIF (AIFu x D) "
            )
        if idx == 0:
            plt.title(SAVE_PATH)
        else:
            plt.legend()
        plt.xlabel("Time [min]")
        plt.ylabel("Activity")
    plt.show()
    if SAVE_FIG:
        print(SAVE_PATH)
        save_path = SAVE_PATH + "AIFs.pdf"
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    # %% Save csv file with corrected data
    if SAVE_CSV:
        save_path = SAVE_PATH + "_PY_disp_corr.csv"
        df2 = pd.DataFrame(
            {
                "Time [min]": t,
                "Measured AIF (AIFm)": AIFm,
                "AIFm_fit": AIFm_fit,
                "Undispersed AIF (AIFu)": AIFu_optimal,
                "Dispersed AIF (AIFu x D)": AIFd_optimal,
            }
        )
        df2.to_csv(save_path, index=False)

    return AIFm_fit_optimal, AIFu_optimal, AIFm_fit_params_optimal


# %% AIF fit with weights


def AIF_fit_weights(t, AIFm, INTERPOLATE=False, TIME_FRAMES=1):
    # Function that fits the Feng1993 AIF function to measured AIF data
    # A fit is performed to the measured data AIFm, and returned as AIFm_fit,
    #
    # This fit function uses a weighted objective function:
    #
    # L(AIFm_fit - AIFm*w)
    #
    # where:
    #
    # AIFm_fit is the fitted curve,
    # AIFm is the measured curve, and
    # w is the weight, w = AIFm^1/n,
    #
    # where
    #
    # n = 1:21
    #
    # This fit is very sensitive to starting parameters (params_init) and to
    # the lower and upper limits (llim, ulim). The suggested values work pretty well.
    #
    #
    # Input:    t                       Time vector in minutes (If seconds are passed, its converted)
    #           AIFm                    The measured AIF
    #
    # Output    AIFm_fit_optimal        The Feng1993 fit to AIFm
    #           AIFfit_params_optimal   The optimal fit parameters
    #
    # Originally written in Matlab by Gustav Kalda & Samuel Kuttner 2018-09-11
    # Translated to Python by Samuel Kuttner 2022-12-06
    #

    # Check if time vector could be in seconds, then convert to minutes
    if np.max(t) > 300:
        t = t / 60

    # Set negative values to zero
    AIFm[AIFm < 0] = 0

    if INTERPOLATE:
        AIFm, _, t_int = interpolate_time_frames(
            AIFm, t * 60, AIFm, t * 60, TIME_FRAMES
        )
        t_org = t
        t = t_int / 60

    # Fit the measured AIF: AIFm
    params_init = [0, 141, 0.296, 0.259, -27, -0.55, -0.04]
    # params_init = [0.975, 80.26, 1, 2.18, -5.235, -0.071, -0.189]
    llim = [0, 0, 0, 0, -100, -1, -1]
    ulim = [3, 5100, 999, 99, 9, 9, 0]
    # ulim = [1.5, 5100, 999, 99, 9, 9, 0] #Original weights from Matlab

    J_max = []  # List to store the max values
    AIFfit_params_list = []
    AIFm_fit_list = []

    for n in range(1, 21):
        weight = np.power(AIFm, 1 / n)

        AIFm_fit_params, pcov, infodict, mesg, ier = curve_fit(
            feng_aif,
            t,
            np.multiply(AIFm, weight),
            method="trf",
            bounds=(llim, ulim),
            p0=params_init,
            maxfev=2**14,
            full_output=True,
        )
        AIFfit_params_list.append(AIFm_fit_params)
        TAUm, A1m, A2m, A3m, L1m, L2m, L3m = AIFm_fit_params
        AIFm_fit = feng_aif(t, TAUm, A1m, A2m, A3m, L1m, L2m, L3m)
        # plt.plot(AIFm_fit)
        # plt.show()
        AIFm_fit_list.append(AIFm_fit)
        J_max.append(((np.max(AIFm_fit) - np.max(AIFm)) ** 2))

    # Find optimal nthroot, as index of minimum J
    optimal_n = np.argmin(J_max)
    AIFfit_params_optimal = AIFfit_params_list[optimal_n]
    AIFm_fit_optimal = AIFm_fit_list[optimal_n]

    if INTERPOLATE:
        # Interpolate to original time frames
        AIFm_fit_optimal = np.interp(t_org, t_int / 60, AIFm_fit_optimal)

    return AIFm_fit_optimal, AIFfit_params_optimal


# %% Supporting functions
def feng_aif(t, TAU, A1, A2, A3, L1, L2, L3):
    # Check if time vector could be in seconds, then convert to minutes
    if np.max(t) > 60:
        t = t / 60
    Cp = (
        (A1 * (t - TAU) - A2 - A3) * np.exp(L1 * (t - TAU))
        + A2 * np.exp(L2 * (t - TAU))
        + A3 * np.exp(L3 * (t - TAU))
    )
    Cp[: np.sum(t < TAU)] = 0
    return Cp


def r_squared(y, yfit):
    # From http://www.graphpad.com/guides/prism/6/curve-fitting/index.htm?reg_diagnostics_tab_7_2.htm

    SSresiduals = np.sum((y - yfit) ** 2)
    SStotal = np.sum(y - np.mean(y) ** 2)

    return 1 - SSresiduals / SStotal


def dispersion_fcn(t, TAUd):
    D = (1 / TAUd) * np.exp(-t / TAUd)
    AUC = np.trapz(D, t)
    D = D / AUC
    return D


def convolve_AIF(A, B, t):
    C = np.convolve(B, A, mode="full") * (t[1] - t[0])
    C = C[0 : len(t)]
    return C


def AIFd_fcn(t, TAU, A1, A2, A3, L1, L2, L3, TAUd):

    AIFd = convolve_AIF(
        feng_aif(t, TAU, A1, A2, A3, L1, L2, L3), dispersion_fcn(t, TAUd), t
    )
    return AIFd


def interpolate_time_frames(C_A, t_A, C_B, t_B, frame_length):
    # Interpolate C_A and C_B to uniform time frames of frame_length [s]
    # Make them equally long by shortening the longest one

    # Create equally spaced target time frames of the shortest sequence
    if t_A[-1] < t_B[-1]:
        t_int = np.linspace(t_A[0], t_A[-1], int(np.round(t_A[-1] / frame_length)))
    else:
        t_int = np.linspace(t_B[0], t_B[-1], int(np.round(t_B[-1] / frame_length)))

    C_A_int = np.interp(t_int, t_A, C_A)
    C_B_int = np.interp(t_int, t_B, C_B)

    return C_A_int, C_B_int, t_int


# %%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# %%
def auc_peak_tail(t, y, t_min, t_break):
    """
    Calculates AUC under peak (t=t_min:t_break) and tail (t=t_break:) as tuple for vector y with time steps t.
    """
    auc_peak = auc(t[t_min:t_break], y[t_min:t_break])
    auc_tail = auc(t[t_break:], y[t_break:])
    return auc_peak, auc_tail


# %% Input function feature extraction


def IF_feature_extract(
    IF_t,
    IF_A,
    DO_PLOT=False,
    YMIN=None,
    YMAX=None,
    SAVE=False,
    SAVE_FIG_PATH=None,
    mouse_name=None,
    UNITS_LABEL=None,
    IF_name=None,
):
    # Extract features from an input function
    # Input: Time vector and Activity vector of input function + supporting parameters
    # Output: Dictionary of features

    features = {}

    # % Peak value [UNITS]
    peak = np.max(IF_A)

    # % Time to peak (ttp) [s]
    ttp = IF_t[np.argmax(IF_A)] * 60

    # % Area under peak and tail. Must start and breakpoint.

    # For IDIF (with only 41 time frames), we must interpolate to uniform spacing
    if len(IF_t) < 50:
        IF_A, _, IF_t = interpolate_time_frames(IF_A, IF_t * 60, IF_A, IF_t * 60, 1)
        IF_t = IF_t / 60

    # Find number of frames for 10s window
    wdw = int(np.round(10 / ((IF_t[-1] - IF_t[-2]) * 60), 0))

    # Smooth the IF by moving average filter
    IF_A2 = np.convolve(IF_A, np.ones(wdw) / wdw, mode="same")

    # Find the first derivative and smooth
    IF_A_grad = np.gradient(np.convolve(IF_A2, np.ones(wdw) / wdw, mode="same"))
    IF_A_grad = IF_A_grad / np.max(IF_A_grad)  # Normalize for easier plotting

    # Second derivative
    IF_A_grad_grad = np.gradient(
        np.convolve(IF_A_grad, np.ones(wdw) / wdw, mode="same")
    )
    IF_A_grad_grad = IF_A_grad_grad / np.max(IF_A_grad_grad)  # Normalize

    # Find max and min points
    IF_A_max = np.max(IF_A2)
    signal = np.mean(IF_A2[:30])
    noise = np.std(IF_A2[:30])

    # This describes an objective and reproducible way to define the break point
    # between peak and tail. Use "AUC_AIF_ID.pdf" plots for understanding:

    # Find break-point between peak and tail as the second peak of the second
    # derivative, which is the equal to the first peak after the AIF peak.
    # To this, add half of the difference between the second peak of the second
    # derivative, and the min-point of the first derivative. This is the break point.

    max_idx = np.argmax(IF_A2)  # Max of IF
    peak_start = find_nearest(IF_A2[0:max_idx], 2 * (signal + noise))
    IF_A_grad_min = np.argmin(IF_A_grad)
    peak_end = (
        np.argmax(IF_A_grad_grad[max_idx:]) + max_idx
    )  # Second peak of second derivative
    peak_end2 = peak_end + int((peak_end - IF_A_grad_min) / 2)

    # Find AUC peak and tail
    auc_peak, auc_tail = auc_peak_tail(IF_t, IF_A, peak_start, peak_end2)

    auc_peak_tail_ratio = auc_peak / auc_tail

    # % FWHM of peak
    fwhm_start = find_nearest(IF_A[:max_idx], IF_A_max / 2)
    fwhm_end = find_nearest(IF_A[max_idx:peak_end2], IF_A_max / 2) + max_idx

    fwhm = (fwhm_end - fwhm_start) * (IF_t[-1] - IF_t[-2]) * 60  # FWHM in seconds

    features = {
        "Peak": peak,
        "TTP": ttp,
        "FWHM": fwhm,
        "AUC peak": auc_peak,
        "AUC tail": auc_tail,
        "AUC ratio": auc_peak_tail_ratio,
    }

    units = {
        "Peak": "[g/ml]",
        "TTP": "[s]",
        "FWHM": "[s]",
        "AUC peak": "[g/ml*min]",
        "AUC tail": "[g/ml*min]",
        "AUC ratio": "[1/1]",
    }

    # %% Plot AUC curves
    if DO_PLOT:

        YMIN = -1.5
        X_axis = (-1, 5)  # minutes
        fig, ax = plt.subplots(figsize=(10, 7))
        for idx, t_max_minutes in enumerate(X_axis):

            if idx == 1:
                inset_axes(
                    ax,
                    width="100%",  # width = 30% of parent_bbox
                    height="100%",  # height : 1 inch
                    loc=1,
                    bbox_to_anchor=(0.45, 0.5, 0.5, 0.5),
                    bbox_transform=ax.transAxes,
                )
                tmax = find_nearest(IF_t, t_max_minutes)
                # t_max_fit = find_nearest(T_IDIF_fit, t_max_minutes)

            else:
                tmax, t_max_fit = -1, -1

            # tmax = 300
            # plt.plot(AIF_t_int[0:tmax], aif_A_int[0:tmax], label="AIF")
            # plt.plot(AIF_t_int[0:tmax], idif_A_int[0:tmax], label="IDIF")
            # plt.plot(AIF_t_int[0:tmax], AIF_A_int_shift[0:tmax], label="AIF_shift")
            # plt.legend()
            # plt.title("Mouse {}".format(mouse_name))
            # plt.show()

            plt.plot(IF_t[:tmax], IF_A[:tmax], label="AIF")
            plt.fill_between(
                IF_t[:tmax],
                IF_A[:tmax],
                0,
                where=(IF_t[:tmax] > IF_t[peak_start])
                & (IF_t[:tmax] <= IF_t[peak_end2]),
                color="g",
                alpha=0.5,
            )
            plt.fill_between(
                IF_t[:tmax],
                IF_A[:tmax],
                0,
                where=(IF_t[:tmax] > IF_t[peak_end2]),
                color="c",
                alpha=0.5,
            )
            plt.plot(IF_t[:tmax], IF_A_grad[:tmax], label="D(AIF)")
            plt.plot(IF_t[:tmax], IF_A_grad_grad[:tmax], label="D$^2$(AIF)")

            plt.plot(
                (IF_t[fwhm_start], IF_t[fwhm_end]),
                (IF_A_max / 2, IF_A_max / 2),
                linewidth=2,
                label="FWHM",
            )

            if idx == 0:
                plt.title(
                    mouse_name
                    + " / Peak: "
                    + str(round(peak, 2))
                    + " / TTP: "
                    + str(round(ttp, 2))
                    + "s"
                    + " / FWHM: "
                    + str(round(fwhm, 2))
                    + "s"
                    + " / AUC$_{peak}$: "
                    + str(round(auc_peak, 2))
                    + " / AUC$_{tail}$: "
                    + str(round(auc_tail, 2))
                    + " / AUC$_{\%peak/tail}$: "
                    + str(round(auc_peak_tail_ratio * 100, 2))
                )
            else:
                plt.legend()

            plt.xlabel("Time [min]")
            plt.ylabel("Activity concentration " + UNITS_LABEL)
            plt.ylim(YMIN, YMAX)

        plt.show()
        if SAVE:
            save_path = SAVE_FIG_PATH + "AUC_" + IF_name + "_" + mouse_name + ".pdf"
            fig.savefig(save_path, bbox_inches="tight")
        return features, units


# %% Orthogonal regression function. From:
#   http://blog.rtwilson.com/orthogonal-distance-regression-in-python/


def orthoregress(x, y):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c, nan, nan, nan]
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.
    """

    def f(p, x):
        """Basic linear regression 'model' for use with ODR"""
        return (p[0] * x) + p[1]

    linreg = stats.linregress(x, y)
    mod = Model(f)
    dat = Data(x, y)
    od = ODR(dat, mod, beta0=linreg[0:2])
    out = od.run()

    return list(out.beta)


# %% Scatterplot
def scatterplot(
    y,
    Y,
    xlabel,
    ylabel,
    title,
    legend_label,
    axis_ticks_min_max_interval=None,
    errorbar=None,
    mouse_IDs=None,
    letter=None,
):
    """
    Plot AUC peak and tail for AIF and MLIF. Calculate comparative measures.

    Parameters
    ----------
    y : List of numpy arrays, each with length T.
        Ground truth AIFs.
    Y : List of numpy arrays, each with length T.
        Predicted MLIFs.


    Returns
    -------
    fig : Figure
        Scatter plot of Y vs y.


    """

    # Check if both y and Y have nans in the same place
    if not all(np.isnan(y) == np.isnan(Y)):
        print("Nans are unequal. Exiting.")
        return

    # Remove nan values if both in y and Y
    # y = y[~np.isnan(y)]
    # Y = Y[~np.isnan(Y)]
    if errorbar is not None:
        errorbar = errorbar[~np.isnan(errorbar)]

    ####################### Start plotting #######################
    # Define some measures for the plot
    lw = 3  # Line width
    mrkrsz = 60  # Marker size
    fs = 14  # Font size (original: 14)
    ls = 18  # Label size

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    x1 = y
    y1 = Y

    ###### T-test & correlation coefficient
    _, p1 = stats.ttest_rel(x1, y1)
    corrcoef1 = np.corrcoef(x1, y1)[0, 1]

    ##############
    # Tried to divide the data into scan1 / scan 2
    ax1.scatter(x1, y1, label=legend_label, color="#e43d40", s=mrkrsz)
    if errorbar is not None:
        ax1.errorbar(x1, y1, yerr=errorbar, capsize=5, fmt="None", ecolor="k")

    if mouse_IDs is not None:
        for i, txt in enumerate(mouse_IDs):
            ax1.annotate(txt, (x1[i], y1[i]))

    range1 = np.max(x1) - np.min(x1)
    range2 = np.max(y1) - np.min(y1)
    range_max = np.max((range1, range2))
    axmin1 = np.min((np.min(x1), np.min(y1))) - range_max / 10
    axmax1 = np.max((np.max(x1), np.max(y1))) + range_max / 10

    # axmin1 = np.min((x1,y1))

    ####### Linear regression line #######
    #            ax1.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))    #Least square regression line

    ###### Orthogonal regression line #######
    beta1 = orthoregress(x1, y1)
    r2_1 = r2_score(np.unique(x1), beta1[0] * np.unique(x1) + beta1[1])

    ax1.plot(
        np.unique([axmin1, axmax1]),
        beta1[0] * np.unique([axmin1, axmax1]) + beta1[1],
        label="Orth. regr.",
        color="#e43d40",
        linewidth=lw,
    )  # Orthogonal regression line
    ax1.plot(
        np.unique([axmin1, axmax1]),
        np.unique([axmin1, axmax1]),
        "--",
        color="black",
        label="y=x",
        linewidth=lw,
    )  # y=x line

    # Change the order of legend items
    handles, labels = ax1.get_legend_handles_labels()
    order = [2, 0, 1]
    labels2 = [labels[i] for i in order]
    handles2 = [handles[i] for i in order]
    legend = ax1.legend(
        handles2,
        labels2,
        loc="lower right",
        prop={"size": fs},
        labelspacing=0.3,
        edgecolor="k",
    )  # , bbox_to_anchor=(1,0.5))
    frame = legend.get_frame()
    frame.set_linewidth(3)

    # Set same tick marks on x and y axes
    start, end = ax1.get_xlim()
    # ax1.xaxis.set_ticks(np.arange(np.round(start/10,0)*10-1, end+1, 0.25))
    # ax1.yaxis.set_ticks(np.arange(np.round(start/10,0)*10-1, end+1, 0.25))

    if axis_ticks_min_max_interval:
        ax1.xaxis.set_ticks(
            np.arange(
                axis_ticks_min_max_interval[0],
                axis_ticks_min_max_interval[1],
                axis_ticks_min_max_interval[2],
            )
        )
        ax1.yaxis.set_ticks(
            np.arange(
                axis_ticks_min_max_interval[0],
                axis_ticks_min_max_interval[1],
                axis_ticks_min_max_interval[2],
            )
        )

    # Set xlim and ylim
    ax1.set_xlim(axmin1, axmax1)
    ax1.set_ylim(axmin1, axmax1)

    # Add text
    if letter:
        ax1.text(axmin1 - range1 / 10, axmax1, letter, weight="bold", size=20)

    ax1.text(
        axmin1 + 0.3 * range1 / 10,
        axmax1 - 1 * range1 / 10,
        f"y = {beta1[0]:.2f}x + {beta1[1]:.2f}",
        size=fs,
    )  # Orthogonal regression line
    ax1.text(
        axmin1 + 0.3 * range1 / 10,
        axmax1 - 2 * range1 / 10,
        f"r$^2$ = {r2_1:.2f}",
        size=fs,
    )
    ax1.text(
        axmin1 + 0.3 * range1 / 10, axmax1 - 3 * range1 / 10, f"P = {p1:.2f}", size=fs
    )
    ax1.text(
        axmin1 + 0.3 * range1 / 10,
        axmax1 - 4 * range1 / 10,
        f"Corr. coeff. = {corrcoef1:.2f}",
        size=fs,
    )

    ax1.set_xlabel(xlabel, size=ls)
    ax1.set_ylabel(ylabel, size=ls)

    ax1.set_title(title, size=fs)

    # Set axis related measures
    ax1.xaxis.set_tick_params(width=lw)
    ax1.yaxis.set_tick_params(width=lw)

    ax1.tick_params(labelsize=ls)

    plt.setp(ax1.spines.values(), linewidth=lw)

    # fig.tight_layout()
    plt.show()

    return fig, corrcoef1
