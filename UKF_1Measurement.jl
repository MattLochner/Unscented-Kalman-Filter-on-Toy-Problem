using DifferentialEquations
using Distributions
using Plots
using LinearAlgebra
using Random
Random.seed!(4)

function Model_prediction(tStart, tEnd, xSigma, p)
    prob3 = ODEProblem(ModelSystem!, xSigma, (tStart, tEnd), p)
    ODEsol2 = solve(prob3, Tsit5(), saveat = tEnd)
    return ODEsol2.u[2]
end

function Measurement_prediction(x, p)
    y = x[3]
    return y                        #This must be done so that y is matrix
end

function UnscentedTransformation(xhatmin1, Pposmin1, nx, ny, R, Q, params, tstart, tend) #nx is number of state variables, ny is number of measurement variables

    #Calculate sigma point weights

    k = nx - 3
    Weights = zeros(2*nx+1)
    Weights[1] = k /(nx + k)
    for i = 2:(2*nx+1)
        Weights[i] = 1/(2*(nx+k))
    end

    #Generate sigma points

    sqroot_Pposmin1 = cholesky((nx+k)*Pposmin1)
    
    xsigma = zeros(size(Pposmin1)[1], (2*nx+1))
    xsigma[:,1] = xhatmin1              #The first sigma point is equal to the posterior state estimate from the previous timestep xhatpos
    for i = 2:(nx+1)
            xsigma[:, i] = xhatmin1 + sqroot_Pposmin1.U[(i-1), :]
            xsigma[:, (i+nx)] = xhatmin1 - sqroot_Pposmin1.U[(i-1), :]
    end

    # Transform the sigma points in the prediction step

    xsigmaneg = zeros(nx, 2*nx+1)             
        
    for i = 1:(2*nx+1)
        xsigmaneg[:,i] = Model_prediction(tstart, tend, xsigma[:,i], params) #change depending on number of x read
    end
    
    # Calculate the prior state estimate from the weighted sum of the prior sigma points

    xhatneg = zeros(nx, 1)

    for i = 1:(2*nx+1)
        xhatneg = xhatneg + Weights[i] * xsigmaneg[:,i]      #Prior state estimate
    end

    # Calculate prior estimation error covariance

    Pneg = zeros(nx, nx)
    check1 = reshape(xsigmaneg[:,1], length(xhatneg), 1)
    check2 = xsigmaneg[:,1]
    check3 = xhatneg

    for i = 1: (2*nx+1)
        Pneg = Pneg + Weights[i] * (reshape(xsigmaneg[:,i], length(xhatneg), 1)-xhatneg) * (reshape(xsigmaneg[:,i], length(xhatneg), 1)-xhatneg)'  #Make sure xhatneg is the right shape
    end
    Pneg = Pneg + Q                 #Prior state estimation error covariance 

    #Generate new sigma points for measurement prediction step

    root_update = cholesky(nx*Pneg)

    xsigma_update = zeros(nx, 2*nx + 1)
    xsigma_update[:,1] = xhatneg        #First sigma point is equal to the prior state estimate

    for i = 2:nx+1
        xsigma_update[:,i] = xhatneg + root_update.U[i-1,:]
        xsigma_update[:,nx+i] = xhatneg - root_update.U[i-1,:]
    end

    #Transform the sigma points using the measurement function

    ysigma = zeros(ny, 2*nx+1)

    for i = 1:(2*nx+1)
            ysigma[i] = Measurement_prediction(xsigma_update[:,i], params) #change depending on 1 or many y
    end

    #Calculate the approximate mean of the predicted measurements using the weighted sum of the transformed sigma points

    yhat = zeros(ny, 1)

    for i = 1:(2*nx+1)
         yhat = yhat + Weights[i] * ysigma[:,i] #Note: if more there is more than one measurement use ysigma[:,i]. Find a better way to generate a matrix than the ;; shortcut (I should write an if else statement to account for both situations)
    end

    #Calculate the approximate covariance of the predicted measurements

    Py = zeros(ny, ny)
    Pxy = zeros(nx, 1)
   
    for i = 1:(2*nx+1)
        Py = Py + Weights[i] * (ysigma[:,i:i] - yhat) * (ysigma[:,i:i] - yhat)' #change logged
        Pxy = Pxy + Weights[i] * (xsigma_update[:,i:i] - xhatneg) * (ysigma[:,i:i] - yhat)'
    end

    #Add measurement noise to measurement covariance matrix
    Py = Py + R

    return xhatneg, Pneg, yhat, Py, Pxy
end

function StateEstimateUpdate(xhatneg, Pneg, yhat, Py, Pxy, y)
    Kk = Pxy * inv(Py)
    xhatpos = xhatneg + Kk * (y - yhat)
    Ppos = Pneg - Kk * Py * Kk'
    Ppos = (Ppos + Ppos') ./ 2 #stabilise for numerical issues
    return xhatpos, Ppos
end

function RunFilter(estimationlength, N_meas, xhat_0, P0, nx, ny, R, Q, y, params)

    # Generate the storage matrices

    xhatArray = [xhat_0]    # array of state estimates
    PArray = [diag(P0)]     # Array of state estimate error covariance (vector form) interesting that it starts with diag(P0)

    # Initialize the state estimator

    xhatpos = xhat_0        # Initial state estimate
    Ppos = P0               # Initial state estimation error covariance
    nx = length(xhat_0)     # Number of state variables
    ny = length(y[1])       # Number of measurement variables

    # Calculate the state estimation at each timestep

    timesteps = 0:N_meas:estimationlength*N_meas #Define the time points in seconds at which the state estimates are calculated based on the measurement sampling rate(N_meas)

    for j = 1:estimationlength
        tStart = timesteps[j]
        tEnd = timesteps[j+1]
        #display(j)
        (xhatneg, Pneg, yhat, Py, Pxy) = UnscentedTransformation(xhat_0, P0, nx, ny, R, Q, params, tStart, tEnd)

        (xhatpos, Ppos) = StateEstimateUpdate(xhatneg, Pneg, yhat, Py, Pxy, y[:, (j+1):(j+1)])
        xhat_0 = xhatpos
        push!(xhatArray, vec(xhatpos))
        P0 = Ppos
    end
    return xhatArray, PArray
end

function GenerateMeasurements(ODEsol, estimationlength, ny, xhat_0, vk_std, measured_variable)
    xvalues = [x[measured_variable] for x in ODEsol.u[1:end]] 
    y = zeros(ny,estimationlength+1)
    y[1] = xhat_0[1]
    for i in 2:(estimationlength+1) #skips first measurement which is the orginal state estimate, measurements start at the first measurment time step.
        y[i] = xvalues[i] + vk_std*randn()
    end
    return y
end

