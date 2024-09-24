import numpy as np
from scipy.optimize import curve_fit
from scipy import stats


def mse_func(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


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


# %% #One tissue compartment model
def oneTCM(Cp, t_p, Ct, t_t=0, INTERPOLATE=True, TIME_FRAMES=1):

    # If t_t is not passed, assume it equals t_p
    if np.sum(t_t) == 0:
        t_t = t_p

    # Interpolate to uniform time framing of 1s
    if INTERPOLATE:
        Cp_int, Ct_int, t_int = interpolate_time_frames(Cp, t_p, Ct, t_t, TIME_FRAMES)
    else:
        t_int = t_p
        Cp_int = Cp
        Ct_int = Ct

    # Convert to minutes
    t_int = t_int / 60

    # Define solution to 1TCM diff equation (no vB)
    # def CM(X, K1, k2):
    # #    K1 = 0.5
    # #    k2 = 0.8
    # #    t2 = t_avg[pat,org,]
    #     Cp, t2 = X
    #     h = K1 * np.exp(-k2*t2)
    #     Ct = np.convolve(h,Cp, mode='full')*(t2[1]-t2[0])
    #     Ct = Ct[0:len(t2)]
    #     return Ct

    # Curve fit no vB
    #    K_init = 0.5, 0.035
    #    K_fit = curve_fit(CM, (Cp_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    #    fit_int = CM((Cp_int, t_int), K_fit[0][0], K_fit[0][1])

    # Curve fit with vB
    K_init = 0.5, 0.035, 0.05
    CM_vB = CM_vB_wrap(2)
    K_fit = curve_fit(
        CM_vB, (Cp_int, t_int), Ct_int, method="trf", bounds=(0, 10), p0=K_init
    )
    #    K_fit = curve_fit(CM_vB, (Cp_int, t_int), Ct_int, method='lm', p0=K_init)
    fit_int = CM_vB((Cp_int, t_int), K_fit[0][0], K_fit[0][1], K_fit[0][2])

    # Interpolate back to original spacing
    fit = np.interp(t_p / 60, t_int, fit_int)

    # K_fit_pred = curve_fit(CM_vB, (Cp_pred_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    return K_fit[0], fit


# %% Irreversible two tissue compartment model (k4=0)
def twoTCMirrev(Cp, t_p, Ct, t_t=0, INTERPOLATE=True, TIME_FRAMES=2.5):

    # Set negative and small positive values to zero in Cp:
    Cp[Cp < 10e-5] = 0

    # If t_t is not passed, assume it equals t_p
    if np.sum(t_t) == 0:
        t_t = t_p

    # Interpolate to uniform time framing of 1s
    if INTERPOLATE:
        Cp_int, Ct_int, t_int = interpolate_time_frames(Cp, t_p, Ct, t_t, TIME_FRAMES)
    else:
        t_int = t_p
        Cp_int = Cp
        Ct_int = Ct

    # Check if time vector could be in seconds, then convert to minutes
    if np.max(t_int) > 60:
        t_int = t_int / 60

    # Cp_pred_int = np.interp(t_int, t2, Cp_pred)

    # Define solution to 1TCM diff equation (no vB)
    # def CM(X, K1, k2):
    # #    K1 = 0.5
    # #    k2 = 0.8
    # #    t2 = t_avg[pat,org,]
    #     Cp, t2 = X
    #     h = K1 * np.exp(-k2*t2)
    #     Ct = np.convolve(h,Cp, mode='full')*(t2[1]-t2[0])
    #     Ct = Ct[0:len(t2)]
    #     return Ct

    # Curve fit no vB
    #    K_init = 0.5, 0.035
    #    K_fit = curve_fit(CM, (Cp_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    #    fit_int = CM((Cp_int, t_int), K_fit[0][0], K_fit[0][1])

    # Curve fit with vB
    # K_init = 0.5, 0.035, 0.035, 0.05
    K_init = 0.1, 0.1, 0, 0.1
    CM_vB = CM_vB_wrap(3)
    K_fit = curve_fit(
        CM_vB,
        (Cp_int, t_int),
        Ct_int,
        method="trf",
        bounds=(0, 10),
        p0=K_init,
        maxfev=5000,
    )
    #    K_fit = curve_fit(CM_vB, (Cp_int, t_int), Ct_int, method='lm', p0=K_init)
    fit_int = CM_vB((Cp_int, t_int), K_fit[0][0], K_fit[0][1], K_fit[0][2], K_fit[0][3])

    # Interpolate back to original spacing
    fit = np.interp(t_p, t_int, fit_int)

    # Calculate net influx rate constant
    Ki = K_fit[0][0] * K_fit[0][3] / (K_fit[0][1] + K_fit[0][3])

    # K_fit_pred = curve_fit(CM_vB, (Cp_pred_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    return K_fit[0], Ki, fit, mse_func(Ct_int, fit_int)


# %% Reversible two tissue compartment model
def twoTCMrev(Cp, t_p, Ct, t_t=0, INTERPOLATE=True, TIME_FRAMES=2.5):

    # Set negative and small positive values to zero in Cp:
    Cp[Cp < 10e-5] = 0

    # If t_t is not passed, assume it equals t_p
    if np.sum(t_t) == 0:
        t_t = t_p

    # Interpolate to uniform time framing of 1s
    if INTERPOLATE:
        Cp_int, Ct_int, t_int = interpolate_time_frames(Cp, t_p, Ct, t_t, TIME_FRAMES)
    else:
        t_int = t_p
        Cp_int = Cp
        Ct_int = Ct

    # Check if time vector could be in seconds, then convert to minutes
    if np.max(t_int) > 60:
        t_int = t_int / 60

    # Curve fit no vB
    #    K_init = 0.5, 0.035
    #    K_fit = curve_fit(CM, (Cp_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    #    fit_int = CM((Cp_int, t_int), K_fit[0][0], K_fit[0][1])

    # Curve fit with vB
    # K_init = 0.5, 0.035, 0.035, 0.05
    K_init = 0.1, 0.1, 0, 0.1, 0.1
    CM_vB = CM_vB_wrap(4)
    K_fit = curve_fit(
        CM_vB,
        (Cp_int, t_int),
        Ct_int,
        method="trf",
        bounds=(0, 10),
        p0=K_init,
        maxfev=5000,
    )
    #    K_fit = curve_fit(CM_vB, (Cp_int, t_int), Ct_int, method='lm', p0=K_init)
    fit_int = CM_vB(
        (Cp_int, t_int), K_fit[0][0], K_fit[0][1], K_fit[0][2], K_fit[0][3], K_fit[0][4]
    )

    # Interpolate back to original spacing
    fit = np.interp(t_p, t_int, fit_int)

    # Calculate net influx rate constant
    Ki = K_fit[0][0] * K_fit[0][3] / (K_fit[0][1] + K_fit[0][3])

    # K_fit_pred = curve_fit(CM_vB, (Cp_pred_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    return K_fit[0], Ki, fit, mse_func(Ct_int, fit_int)


# %%
def CM_vB_wrap(how_many_k=2):
    def CM_vB(X, K1, k2, vB, *args):
        Cp, t2 = X
        if how_many_k == 2:
            h = K1 * np.exp(-k2 * t2)
        else:
            if how_many_k == 3:
                k3 = args[0]
                h = K1 / (k2 + k3) * (k3 + k2 * np.exp(-(k2 + k3) * t2))
            else:
                k3 = args[0]
                k4 = args[1]
                a1 = 0.5 * (k2 + k3 + k4 + np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4))
                a2 = 0.5 * (k2 + k3 + k4 - np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4))

                h = (
                    K1
                    / (a1 - a2)
                    * (
                        (a1 - k3 - k4) * np.exp(-(a1 * t2))
                        - (a2 - k3 - k4) * np.exp(-(a2 * t2))
                    )
                )

        Ct = np.convolve(h, Cp, mode="full") * (t2[1] - t2[0])
        Ct = Ct[0 : len(t2)]

        Ct = (1 - vB) * Ct + vB * Cp

        return Ct

    return CM_vB


# %% Patlak model
def patlak(Cp, t_p, Ct, t_t=0, INTERPOLATE=True, TIME_FRAMES=2.5):
    """
    PET Patlak modeling
    Input:  Cp, Plasma time-activity curve
            Ct, Tissue time-activity curve
            t_p, Time vector for plasma curve in seconds
            t_t, Time vector for tissue curve in seconds. If none, it equals t_p
            INTERPOLATE, Perform interpolation or not. Default is 2.5 second frames. Call with False if interpolation is done outslide this function.
    Output: Ki, Net influx, obtained as the slope of the linear fit of the steady-state part of the plot in the transformed space.
            V0, Initial volume of distribution, obtained as the y-axis crossing from the same fit as Ki

    To find the point at which steady state has occured, two linear models are fitted to the early and late parts of the normalized curve.
    The mean squared error (MSE) is calculated for each fit and summed. The break point is taken as the time where the minimum summed MSE is obtained.
    The search for the break point is limited between the second time point and the first 10 minutes of the curve.

    Update 5/10-2022: Set negative and small positive values in Cp to zero in early time frames. Patlak crashes if not!
    """

    # Hyperparmeters
    # timesteps = 1440  # Interpolate to uniform time frames of approx 2.5s
    divisor = 6  # Search for the optimal break point between the two linear curves from t=0 to t=1/divisor of the maximum time. For instnace, if divisor=6, then it searches within first 10 minutes for a 60 minute scan.

    # Set negative and small positive values to zero in Cp:
    Cp[Cp < 10e-5] = 0

    # If t_t is not passed, assume it equals t_p
    if np.sum(t_t) == 0:
        t_t = t_p

    # Interpolate to uniform time framing of 1s
    if INTERPOLATE:
        Cp_int, Ct_int, t_int = interpolate_time_frames(Cp, t_p, Ct, t_t, TIME_FRAMES)
    else:
        t_int = t_p
        Cp_int = Cp
        Ct_int = Ct

    # Check if time vector could be in seconds, then convert to minutes
    if np.max(t_int) > 60:
        t_int = t_int / 60

    def integrate(C, dt):
        return np.cumsum(C * dt)

    def zero_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def mse_func(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    dt = t_int[2] - t_int[1]  # time framing in minutes

    # Create the new axes (normalized time)
    newX = zero_divide(integrate(Cp_int, dt), Cp_int)
    newY = zero_divide(Ct_int, Cp_int)

    # Linear regression. Find the optimal time range for the linear part by fitting two lines on each side of a break point, t_b.
    # Optimal break point is where the summed mean squared error between the two fits to their respective part of the curve is smallest.
    # Net influx (Ki) is the slope of the second curve fit: slope2_final

    mse_final = 999999
    first_nonzero = np.nonzero(newX)[0][0]

    # Remove leading zero elements
    newX = newX[first_nonzero:]
    newY = newY[first_nonzero:]

    # Optimize break point, t_b between two linear fits (0->t_b and t_b->end)
    # Limit the search between the second time point and the first 10 minutes (1/6th of the time points)

    for t_b in range(2, int(len(newX) / divisor)):
        # Fit first part of curve (0->t_b)
        newX1 = newX[0:t_b]
        newY1 = newY[0:t_b]
        slope1, intercept1, r1, p1, std_err1 = stats.linregress(newX1, newY1)
        model1 = slope1 * newX1 + intercept1
        mse1 = mse_func(model1, newY1)

        # Fit second part of curve (t_b->end)
        newX2 = newX[t_b:]
        newY2 = newY[t_b:]
        slope2, intercept2, r2, p2, std_err2 = stats.linregress(newX2, newY2)
        model2 = slope2 * newX2 + intercept2
        mse2 = mse_func(model2, newY2)

        # Combined MSE
        mse = mse1 + mse2

        # Assign new break point if current mse is smaller than any earlier smallest mse
        if t_b == 2 or mse < mse_final:
            t_b_final = t_b
            slope1_final = slope1
            intercept1_final = intercept1
            slope2_final = slope2
            intercept2_final = intercept2
            mse_final = mse
    # # Optional plotting:

    # plt.plot(t_int[0:tmax], Cp[0:tmax])
    # plt.plot(t_int[0:tmax], Ct[0:tmax])
    # plt.show()

    # Plot the final model with the two model fits.
    # model1_final = slope1_final * newX + intercept1_final
    # model2_final = slope2_final * newX + intercept2_final
    # plt.plot(newX, newY, "r", linewidth=3)
    # plt.plot(newX[:t_b_final], model1_final[:t_b_final], "g--")
    # plt.plot(newX, model2_final, "b--")
    # plt.show()
    # print(slope2_final)

    return slope2_final, intercept2_final
