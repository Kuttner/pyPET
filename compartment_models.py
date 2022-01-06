import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

#%% One tissue compartment model
def oneTCM(t, Cp, Ct, UNITS='s',INIT = (0.5, 0.035, 0.05)):
'''
    One tissue compartment model.

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    Cp : TYPE
        DESCRIPTION.
    Ct : TYPE
        DESCRIPTION.
    UNITS : TYPE, optional
        DESCRIPTION. The default is 's'.
    INIT : TYPE, optional
        DESCRIPTION. The default is (0.5, 0.035, 0.05).

    Returns
    -------
    output : The fitted rate constants, where K1 = ouput[0], k2 = ouput[1] and vB = output[-1]

    '''
    
    # Convert to minutes
    if UNITS=='s':
        t2 = t / 60  
    elif UNITS=='min':
        pass
    else:
        print('Unknown time units. Must be either "s" or "min".')
        break
      
    # Interpolate to uniform 1s time framing              
    t_int = np.linspace(t2[0], t2[-1], int(np.round(t2[-1])))
    Cp_int = np.interp(t_int, t2, Cp)
    Ct_int = np.interp(t_int, t2, Ct)
    
    #Define comparment model using wrapper function (comp_mod)
    comp_mod = comp_mod_wrap(2)
    
    #Fit the model to the data
    output = curve_fit(comp_mod, (Cp_int, t_int), Ct_int, method="trf", bounds=(0, 10), p0=INIT)
    
    #Calculate the model tissue curve
    Ct_model_int = comp_mod((Cp_int, t_int), output[0][0], output[0][1], output[0][2])

    # Interpolate back to original spacing
    Ct_model = np.interp(t2, t_int, Ct_model_int)

    return output, Ct_model

#CONTINUE:
    # This function is more or less finished. 
    # Test it, and then try to make it generic, to function as all compartment models + patlak

#%% Two tissue compartment model
def twoTCM(t, Cp, Ct):

    # Interpolate to uniform time framing
    t2 = t / 60  # Convert to minutes
    t_int = np.linspace(t2[0], t2[-1], 3600)  # approx 1s frames
    Cp_int = np.interp(t_int, t2, Cp)
    Ct_int = np.interp(t_int, t2, Ct)
    # Cp_pred_int = np.interp(t_int, t2, Cp_pred)

    # Do not interpolate when using 0-6min raw data, already uniform time samples from AIF blood counter (and PET data is interpolated to same framing!)
    # Cp_int = Cp
    # Ct_int = Ct
    # t_int = t2

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
    #    output = curve_fit(CM, (Cp_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    #    fit_int = CM((Cp_int, t_int), output[0][0], output[0][1])

    # Curve fit with vB
    # K_init = 0.5, 0.035, 0.035, 0.05

    K_init = 0.1, 0.1, 0.1, 0.0
    CM_vB = comp_mod_wrap(3)
    output = curve_fit(
        comp_mod,
        (Cp_int, t_int),
        Ct_int,
        method="trf",
        bounds=(0, 10),
        p0=K_init,
        maxfev=5000,
    )
    #    output = curve_fit(comp_mod, (Cp_int, t_int), Ct_int, method='lm', p0=K_init)
    fit_int = comp_mod((Cp_int, t_int), output[0][0], output[0][1], output[0][2], output[0][3])

    # Interpolate back to original spacing
    fit = np.interp(t2, t_int, fit_int)

    # Calculate net influx rate constant
    Ki = output[0][0] * output[0][2] / (output[0][1] + output[0][2])

    # output_pred = curve_fit(comp_mod, (Cp_pred_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    return output[0], Ki, fit


#%% Reversible two tissue compartment model
def twoTCMrev(t, Cp, Ct):

    # Interpolate to uniform time framing
    t2 = t / 60  # Convert to minutes
    t_int = np.linspace(t2[0], t2[-1], 3600)  # approx 1s frames
    Cp_int = np.interp(t_int, t2, Cp)
    Ct_int = np.interp(t_int, t2, Ct)
    # Cp_pred_int = np.interp(t_int, t2, Cp_pred)

    # Curve fit no vB
    #    K_init = 0.5, 0.035
    #    output = curve_fit(CM, (Cp_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    #    fit_int = CM((Cp_int, t_int), output[0][0], output[0][1])

    # Curve fit with vB
    # K_init = 0.5, 0.035, 0.035, 0.05
    K_init = 0.1, 0.1, 0.1, 0.1, 0.0
    comp_mod = comp_mod_wrap(4)
    output = curve_fit(
        comp_mod,
        (Cp_int, t_int),
        Ct_int,
        method="trf",
        bounds=(0, 10),
        p0=K_init,
        maxfev=5000,
    )
    #    output = curve_fit(comp_mod, (Cp_int, t_int), Ct_int, method='lm', p0=K_init)
    fit_int = comp_mod(
        (Cp_int, t_int), output[0][0], output[0][1], output[0][2], output[0][3], output[0][4]
    )

    # Interpolate back to original spacing
    fit = np.interp(t2, t_int, fit_int)

    # #Plot tissue curve
    # tmax=44
    # # fit_y = fit
    # plt.plot(t[0:tmax], fit[0:tmax])
    # # plt.plot(t[0:tmax], fit_y[0:tmax])
    # plt.plot(t[0:tmax], D[i,VOIs_analyze[j],0:tmax])
    # print(output[0])

    # Calculate net influx rate constant
    Ki = output[0][0] * output[0][2] / (output[0][1] + output[0][2])

    # output_pred = curve_fit(comp_mod, (Cp_pred_int, t_int), Ct_int, method='trf', bounds=(0, 10), p0=K_init)
    return output[0], Ki, fit


def comp_mod_wrap(how_many_k=2):
    def comp_mod(X, K1, k2, vB, *args):
        Cp, t2 = X
        if how_many_k == 2:
            h = K1 * np.exp(-k2 * t2)
        elif how_many_k == 3:
            k3 = args[0]
            h = K1 / (k2 + k3) * (k3 + k2 * np.exp(-(k2 + k3) * t2))
        else:
            k3 = args[0]
            k4 = args[1]
            a1 = 0.5 * (k2 + k3 + k4 + np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4))
            a2 = 0.5 * (k2 + k3 + k4 - np.sqrt((k2 + k3 + k4) ** 2 - 4 * k2 * k4))

            h = (K1 / (a1 - a2) * ( (a1 - k3 - k4) * np.exp(-(a1 * t2))  - (a2 - k3 - k4) * np.exp(-(a2 * t2)) )
            )

        Ct = np.convolve(h, Cp, mode="full") * (t2[1] - t2[0])
        Ct = Ct[0 : len(t2)]

        Ct = (1 - vB) * Ct + vB * Cp

        return Ct

    return comp_mod


# %% Patlak model
def patlak(t, Cp, Ct):

    """
    PET Patlak modeling
    Input:  t, time vector
            Cp, Plasma time-activity curve
            Ct, Tissue time-activity curve
    Output: Ki, Net influx, obtained as the slope of the linear fit of the steady-state part of the plot in the transformed space.
            V0, Initial volume of distribution, obtained as the y-axis crossing from the same fit as Ki

    To find the point at which steady state has occured, two linear models are fitted to the early and late parts of the normalized curve.
    The mean squared error (MSE) is calculated for each fit and summed. The break point is taken as the time where the minimum summed MSE is obtained.
    The search for the break point is limited between the second time point and the first 10 minutes of the curve.
    """

    # Hyperparmeters
    timesteps = 1440  # Interpolate to uniform time frames of approx 2.5s
    divisor = 6  # Search for the optimal break point between the two linear curves from t=0 to t=1/divisor of the maximum time. For instnace, if divisor=6, then it searches within first 10 minutes for a 60 minute scan.

    ####### Use these for manual plotting #######
    # i = 13
    # # i+=1
    # print(i)
    # j = 0
    # Cp = Y[i]
    # Ct = D[i, VOIs_analyze[j]]
    # tmax = 34
    ##############################################

    # Interpolate to uniform time framing of 1s
    t2 = t / 60  # Convert to minutes
    t_int = np.linspace(t2[0], t2[-1], timesteps)
    Cp_int = np.interp(t_int, t2, Cp)
    Ct_int = np.interp(t_int, t2, Ct)

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

    # Optional plotting:
    # plt.plot(t[0:tmax], Cp[0:tmax])
    # plt.plot(t[0:tmax], Ct[0:tmax])
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
