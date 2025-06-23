using DifferentialEquations
using Distributions
using Plots
using LinearAlgebra
using Random
Random.seed!(1)

function ToyProblemNormal!(dx, x, p, t)
    u1, u2, c1, Sp, g, A, c2, c3, alpha, r = p.u1, p.u2, p.c1, p.Sp, p.g, p.A, p.c2, p.c3, p.alpha, p.r
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    qf = 0
    dx[1] = (-c1 * Sp * sign(x1 - x3) * sqrt(2 * g * abs(x1 - x3)) + u1)/(A)
    dx[2] = (-c3 * Sp * sign(x2 - x3) * sqrt(2 * g * abs(x2 - x3)) - c2 * Sp * sqrt(2 * g * x2) - qf + u2)/(A)
    dx[3] = (c1 * Sp * sign(x1 - x3) * sqrt(2 * g * abs(x1 - x3)) - c3 * Sp * sign(x3 - x2) * sqrt(2 * g * abs(x3 - x2)))/(A)
end

mutable struct Params
    u1 #m^3/s
    u2 #m^3/s
    c1 #Normal(1.0, 0.0025)
    Sp #m^2
    g #m/s^2
    A #m^2
    c2 #Normal(0.8, 0.0025)
    c3 #Normal(1.0, 0.0025)
    alpha #Normal(0.6, 4e-4)
    r # Normal(0.002, 10e-6)
end

function ModelSystem(dx, x, p, t)
    #u1, u2, c1, Sp, g, A, c2, c3, alpha, r = p.u1*1.1, p.u2*0.9, p.c1*0.8, p.Sp, p.g, p.A, p.c2*0.9, p.c3*1.2, p.alpha, p.r*1.2
    u1, u2, c1, Sp, g, A, c2, c3, alpha, r = p.u1, p.u2, p.c1, p.Sp, p.g, p.A, p.c2, p.c3, p.alpha, p.r
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    qf = 0
    dx[1] = (-c1 * Sp * sign(x1 - x3) * sqrt(2 * g * abs(x1 - x3)) + u1)/(A)
    dx[2] = (-c3 * Sp * sign(x2 - x3) * sqrt(2 * g * abs(x2 - x3)) - c2 * Sp * sqrt(2 * g * x2) - qf + u2)/(A)
    dx[3] = (c1 * Sp * sign(x1 - x3) * sqrt(2 * g * abs(x1 - x3)) - c3 * Sp * sign(x3 - x2) * sqrt(2 * g * abs(x3 - x2)))/(A)
end

function Model_prediction(tStart, tEnd, xSigma, p)
    prob3 = ODEProblem(ModelSystem, xSigma, (tStart, tEnd), p)
    ODEsol2 = solve(prob3, Tsit5(), saveat = tEnd)
    return ODEsol2.u[2]
end
function Measurement_prediction(x, p)
    y = x[3]
    return y
end

function UKF(estimationlength, N_meas, xhat_0, P0, R, Q, y, params)         #the length of y must be the same as the estimation length.
    

    # Generate the storage matrices
    xhatArray = [xhat_0]    # array of state estimates
    PArray = [diag(P0)]     # Array of state estimate error covariance (vector form)
    innArray = []           # Array of innovation terms

    # Initialize the state estimator

    xhatpos = xhat_0        # Initial state estimate
    Ppos = P0               # Initial state estimation error covariance
    nx = length(xhat_0)     # Number of state variables
    ny = length(y[1])         # Number of measurement variables

    # Generate the sigma point weights

    k = nx - 3
    Ws = zeros(2*nx+1)
    Ws[1] = k /(nx + k)
    for i = 2:(2*nx+1)
        Ws[i] = 1/(2*(nx+k))
    end

    # Calculate the state estimation at each timestep

    timesteps = 0:N_meas:estimationlength*N_meas #Define the time points in seconds at which the state estimates are calculated based on the measurement sampling rate(N_meas)

    for j = 1:estimationlength

        #PREDICTION STEP
        #display("xhatpos")
        #display(xhatpos)
        #Generate the sigma points for the prediction step
        root_pred = cholesky((nx+k)*Ppos)

        #This is a dangerous area for matrix vs vector problems, when troubleshooting make sure this works right

        xsigma = zeros(size(xhatpos)[1], (2*nx+1))
        xsigma[:,1] = xhatpos              #The first sigma point is equal to the posterior state estimate form the previous timestep xhatpos
        for i = 2:(nx+1)
            xsigma[:, i] = xhatpos + root_pred.U[(i-1), :]
            xsigma[:, (i+nx)] = xhatpos - root_pred.U[(i-1), :]
        end

        # Transform the sigma points in the prediction step
       
        xsigmaneg = zeros(nx, 2*nx+1)             
        
        for i = 1:(2*nx+1)
            tStart = timesteps[j]
            tEnd = timesteps[j+1]
            xsigmaneg[:,i] = Model_prediction(tStart, tEnd, xsigma[:,i], params)
        end

        # Calculate the prior state estimate from the weighted sum of the prior sigma points

        xhatneg = zeros(nx, 1)

        for i = 1:(2*nx+1)
            xhatneg = xhatneg + Ws[i] * xsigmaneg[:,i]      #Prior state estimate
        end

        # Calculate prior estimation error covariance

        Pneg = zeros(nx, nx)
        check1 = reshape(xsigmaneg[:,1], length(xhatneg), 1)
        check2 = xsigmaneg[:,1]
        check3 = xhatneg

        for i = 1: (2*nx+1)
            Pneg = Pneg + Ws[i] * (reshape(xsigmaneg[:,i], length(xhatneg), 1)-xhatneg) * (reshape(xsigmaneg[:,i], length(xhatneg), 1)-xhatneg)'  #Make sure xhatneg is the right shape
        end
        Pneg = Pneg + Q                 #Prior state estimation error covariance 
        #display("Pneg")
        #display(Pneg)
        #Calculate the innovation terms (why wasn't the method where you generate sigma points and then propogate used here)
        
        #yhat = Measurement_prediction(xhatneg, params)
        #inn = y[:,j] - yhat #There must be j measurements
        #push!(innArray, inn)
    
        #UPDATE STEP

        #Generate the sigma points for the update step

        root_update = cholesky(nx*Pneg)

        xsigma_update = zeros(nx, 2*nx + 1)
        xsigma_update[:,1] = xhatneg        #First sigma point is equal to the prior state estimate

        for i = 2:nx+1
            xsigma_update[:,i] = xhatneg + root_update.U[i-1,:]
            xsigma_update[:,nx+i] = xhatneg - root_update.U[i-1,:]
        end

        #Transform the sigma points using the measurement function

        ysigma = zeros(2*nx+1)

        for i = 1:(2*nx+1)
            ysigma[i] = Measurement_prediction(xsigma_update[:,i], params)
        end
        
        #Calculate the approximate mean of the predicted measurements using the weighted sum of the transformed sigma points

        yhat = zeros(ny, 1)

        for i = 1:(2*nx+1)
            yhat = yhat + [Ws[i] * ysigma[i];;] #Note: if more there is more than one measurement use ysigma[:,i]. Find a better way to generate a matrix than the ;; shortcut (I should write an if else statement to account for both situations)
        end

        #Calculate the approximate covariance of the predicted measurements

        Py = [0;;];
        Pxy = zeros(nx, 1)

        check = Py + Ws[1] * ([ysigma[1];;] - yhat) * ([ysigma[1];;] - yhat)

        for i = 1:(2*nx+1)
            Py = Py + Ws[i] * ([ysigma[i];;] - yhat) * ([ysigma[i];;] - yhat) #Note: if more there is more than one measurement use ysigma[:,i]. Will probably have to use matrix and transpose matrix then (I should write an if else statement to account for both situations)
            Pxy = Pxy + Ws[i]* (xsigma_update[:,i] - xhatneg) * ([ysigma[i];;] - yhat)'
        end
        
        Py = Py + R

        #display(Py)
        #display("Pxy")
        #display(Pxy)

        #Kalman update

        Kk = Pxy * inv(Py)
        xhatpos = xhatneg + Kk * ([y[j+1];;] - yhat) #first measurement in j is the initial state and not the measurement for the first estimation step
        
        display("y[j]")
        display(y[j+1])
        display("x3")
        display(xhatpos[3])

        Ppos = Pneg - Kk * Py * Kk'
        #display(Pneg)
        #display(Ppos)
        #For stability getting rid of small numerical problems that make the matrices not symmetrical

        Ppos = (Ppos + Ppos') ./ 2
     
        # display("j")
        # display(j)
        # display(Ppos)
        # checkeigs = eigen(Ppos)
        # display("eigs")
        # display(checkeigs)
        # check = cholesky(Ppos)
        # display(check)
        # display("check passed")
        push!(xhatArray, vec(xhatpos))
        #display(PArray)
        #push!(PArray, Ppos) #Why did Isabella take the diagonal here?
    end
    return xhatArray, PArray
end

#Give process parameters and initial conditions

normalparams = Params(6e-5, 6e-5, 1, 5e-5, 9.81, 0.0154, 0.8, 1.0, 0.6, 0.002)

x0 = [0.6, 0.4, 0.45]

#Generate true state conditions

estimationlength = 20      #specify the length of the state estimation period by the number of state estimates obtained at the end of the simulation.
N_meas = 10                #specify the measurement sampling rate
tstart = 0
tend = tstart + estimationlength * N_meas
timesteps = tstart:N_meas:(estimationlength*N_meas)

#Set up ODE problem of "real process"

prob = ODEProblem(ToyProblemNormal!, x0, (tstart, tend), normalparams)
ODEsol = solve(prob, Tsit5(), saveat = timesteps)
ODEsol2 = solve(prob, Tsit5())

#Specify initial conditions for Kalman Filter

xhat_0 = [0.5, 0.5, 0.5]
nx = length(xhat_0)
P0 = 0.0001*I(nx) #initial covariance matrix
Q = 0.0001*I(nx) #process noise covariance matrix
vk_std = 0.02 #standard deviation of measurement noise
R = [vk_std ^2] #measurment noise variance

#Generate measurements
x3values = [x[3] for x in ODEsol.u[1:end]] #skips first measurement which is the orginal state estimate, measurements start at the first measurment time step.
y = zeros(estimationlength+1)
y[1] = xhat_0[1]
x3values
estimationlength
for i in 2:(estimationlength+1)
    y[i] = x3values[i] + vk_std*randn()
end

display("true x3 values")
display(x3values)
display("y values")
display(y)
#plot(timesteps, x1values)
#plot!(timesteps, y)

(xhatarray, PArray) = UKF(estimationlength, N_meas, xhat_0, P0, R, Q, y, normalparams)

xhatarray

x1_stateestimates = [x[1] for x in xhatarray]
x2_stateestimates = [x[2] for x in xhatarray]
x3_stateestimates = [x[3] for x in xhatarray]

x1_truestate = [x[1] for x in ODEsol2.u]
x2_truestate = [x[2] for x in ODEsol2.u]
x3_truestate = [x[3] for x in ODEsol2.u]

plot(timesteps, x1_stateestimates, lab = "x1 est")
display(plot!(ODEsol2.t, x1_truestate, lab = "x1 true"))

plot(timesteps, x2_stateestimates, lab = "x1 est")
display(plot!(ODEsol2.t, x2_truestate, lab = "x2 true"))

plot(timesteps, x3_stateestimates, lab = "x3 est")
plot!(timesteps, y, lab = "x3 measurements")
display(plot!(ODEsol2.t, x3_truestate, lab = "x3 true"))

xhatarray