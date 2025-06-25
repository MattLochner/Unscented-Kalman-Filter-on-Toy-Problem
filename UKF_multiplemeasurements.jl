using DifferentialEquations
using Distributions
using Plots
using LinearAlgebra
using Random
Random.seed!(1)

# Provide ODE function of real life process

function modelinput()
    u = 0
    return u
end 

function pendcart!(dx, x, p, t)
    m, M, L, g, d, s = p.m, p.M, p.L, p.g, p.d, p.s
    x1, x2, x3, x4 = x

    Sx = sin(x3)
    Cx = cos(x3)
    D = m * L * L * (M + m * (1 - Cx^2))
    u = modelinput()

    dx[1] = x2
    dx[2] = (1 / D) * (- m^2 * L^2 * g * Cx * Sx + m * L^2 * (m * L * x4^2 * Sx - d * x2)) + m * L * L * (1 / D) * u
    dx[3] = x4
    dx[4] = (1 / D) * ((m + M) * m * g * L * Sx - m * L * Cx * (m * L * x4 ^ 2 * Sx - d * x2)) - m * L * Cx * (1 / D) * u
end
# Provide mutable struct of parameters for real life process

mutable struct Params5
    m           #pendulum mass
    M           #cart mass
    L           #pendulum length
    g           #gravity
    d           #damping
    s           #pendulum position (up = 1, down = -1)
end
# Provide ODE which serves as model of the real life process

function ModelSystem(dx, x, p, t)
    m, M, L, g, d, s = p.m, p.M, p.L, p.g, p.d, p.s #add some variation to system model
    x1, x2, x3, x4 = x

    Sx = sin(x3)
    Cx = cos(x3)
    D = m * L * L * (M + m * (1 - Cx^2))
    u = modelinput()

    dx[1] = x2
    dx[2] = (1 / D) * (- m^2 * L^2 * g * Cx * Sx + m * L^2 * (m * L * x4^2 * Sx - d * x2)) + m * L * L * (1 / D) * u
    dx[3] = x4
    dx[4] = (1 / D) * ((m + M) * m * g * L * Sx - m * L * Cx * (m * L * x4 ^ 2 * Sx - d * x2)) - m * L * Cx * (1 / D) * u
end

function Model_prediction(tStart, tEnd, xSigma, p)
    prob3 = ODEProblem(ModelSystem, xSigma, (tStart, tEnd), p)
    ODEsol2 = solve(prob3, Tsit5(), saveat = tEnd)
    return ODEsol2.u[2]
end

function Measurement_prediction(x, ny, states)
    y = zeros(ny,1)
    for i in ny
        y[i] = x[states[i]]
    end
    return y                        #This must be done so that y is matrix
end

function UnscentedTransformation(xhatmin1, Pposmin1, nx, ny, R, Q, params, tstart, tend, states) #nx is number of state variables, ny is number of measurement variables

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
        xsigmaneg[:,i] = Model_prediction(tstart, tend, xsigma[:,i], params)
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
        ysigma[:,i] = Measurement_prediction(xsigma_update[:,i], ny, states)    #Change logged
    end

    #Calculate the approximate mean of the predicted measurements using the weighted sum of the transformed sigma points

    yhat = zeros(ny, 1)

    for i = 1:(2*nx+1)
         yhat = yhat + Weights[i] * ysigma[:,i] #Note: if more there is more than one measurement use ysigma[:,i]. Find a better way to generate a matrix than the ;; shortcut (I should write an if else statement to account for both situations)
    end

    #Calculate the approximate covariance of the predicted measurements

    Py = zeros(ny, ny)
    Pxy = zeros(nx, ny)
   
    #check1 = ysigma[:,2:2]
    #check2 = yhat
    #display(check1)
    #display(check2)
    #check3 = check1-check2
    #display(check3)
    #check4 = Py
    #display(Py)

    for i = 1:(2*nx+1)
        Py = Py + Weights[i] * (ysigma[:,i:i] - yhat) * (ysigma[:,i:i] - yhat)' #Change logged
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

function RunFilter(estimationlength, N_meas, xhat_0, P0, nx, ny, R, Q, y, params, states)

    # Generate the storage matrices

    xhatArray = [xhat_0]    # array of state estimates
    PArray = [diag(P0)]     # Array of state estimate error covariance (vector form) interesting that it starts with diag(P0)

    # Initialize the state estimator

    xhatpos = xhat_0        # Initial state estimate
    Ppos = P0               # Initial state estimation error covariance

    # Calculate the state estimation at each timestep

    timesteps = 0:N_meas:estimationlength*N_meas #Define the time points in seconds at which the state estimates are calculated based on the measurement sampling rate(N_meas)

    for j = 1:estimationlength
        tStart = timesteps[j]
        tEnd = timesteps[j+1]
        #display(j)
        (xhatneg, Pneg, yhat, Py, Pxy) = UnscentedTransformation(xhat_0, P0, nx, ny, R, Q, params, tStart, tEnd, states)

        (xhatpos, Ppos) = StateEstimateUpdate(xhatneg, Pneg, yhat, Py, Pxy, y[:, (j+1):(j+1)])
        xhat_0 = xhatpos
        push!(xhatArray, vec(xhatpos))
        P0 = Ppos
    end
    return xhatArray, PArray
end

# #Give process parameters and initial conditions

# normalparams = Params5(1, 5, 2, -10, 1, -1)

# x0 = [-1.0; 0; pi+0.1; 0]

# #Generate true state conditions

# estimationlength = 100      #specify the length of the state estimation period by the number of state estimates obtained at the end of the simulation.
# N_meas = 0.1               #specify the measurement sampling rate
# tstart = 0
# tend = tstart + estimationlength * N_meas
# timesteps = tstart:N_meas:(estimationlength*N_meas)

# #Set up ODE problem of "real process"

# prob = ODEProblem(pendcart!, x0, (tstart, tend), normalparams)
# ODEsol = solve(prob, Tsit5(), saveat = timesteps)
# ODEsol2 = solve(prob, Tsit5())

# #Specify initial conditions for Kalman Filter

# xhat_0 = [-1.1; 0.1; pi+0.15; 0.1]
# nx = length(xhat_0) #number of states
# ny = 2 #number of measurements
# states = [1, 3]
# P0 = 0.01*I(nx) #initial covariance matrix
# Q = 0.01*I(nx) #process noise covariance matrix
# vk_std = 1 #standard deviation of measurement noise
# R = Diagonal(fill(vk_std ^2,ny)) #measurment noise variance

# #Generate measurements
# x1values = [x[1] for x in ODEsol.u[1:end]] 
# x3values = [x[3] for x in ODEsol.u[1:end]]
# y = zeros(ny,estimationlength+1)
# y[1,1] = xhat_0[1]
# y[2,1] = xhat_0[3]
# x1values
# estimationlength
# for i in 2:(estimationlength+1) #skips first measurement which is the orginal state estimate, measurements start at the first measurment time step.
#     y[1,i] = x1values[i] + vk_std*randn()
#     y[2,i] = x3values[i] + vk_std*randn()
# end

# display(y)
# ODEsol2.u

# (xhatarray, PArray) = RunFilter(estimationlength, N_meas, xhat_0, P0, nx, ny, R, Q, y, normalparams, states)

# x1_stateestimates = [x[1] for x in xhatarray]
# x2_stateestimates = [x[2] for x in xhatarray]
# x3_stateestimates = [x[3] for x in xhatarray]
# x4_stateestimates = [x[4] for x in xhatarray]


# x1_truestate = [x[1] for x in ODEsol2.u]
# x2_truestate = [x[2] for x in ODEsol2.u]
# x3_truestate = [x[3] for x in ODEsol2.u]
# x4_truestate = [x[4] for x in ODEsol2.u]
# display(y)
# y_measurements1 = y[1,:]
# y_measurements3 = y[2,:]

# plot(timesteps, x1_stateestimates, lab = "x1 est")
# plot!(timesteps, y_measurements1, lab = "x1 measurements")
# display(plot!(ODEsol2.t, x1_truestate, lab = "x1 true"))

# plot(timesteps, x2_stateestimates, lab = "x1 est")
# display(plot!(ODEsol2.t, x2_truestate, lab = "x2 true"))

# plot(timesteps, x3_stateestimates, lab = "x3 est")
# plot!(timesteps, y_measurements3, lab = "x3 measurements")
# display(plot!(ODEsol2.t, x3_truestate, lab = "x3 true"))

# plot(timesteps, x4_stateestimates, lab = "x4 est")
# display(plot!(ODEsol2.t, x4_truestate, lab = "x4 true"))