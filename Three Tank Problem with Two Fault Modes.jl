using DifferentialEquations
using Distributions
using Plots
using StatsBase

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
function ToyProblemF1!(dx, x, p, t)
    u1, u2, c1, Sp, g, A, c2, c3, alpha, r = p.u1, p.u2, p.c1, p.Sp, p.g, p.A, p.c2, p.c3, p.alpha, p.r
    u1 = u1 + (alpha - 1) * u1 
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    qf = 0 
    dx[1] = (-c1 * Sp * sign(x1 - x3) * sqrt(2 * g * abs(x1 - x3)) + u1)/(A)
    dx[2] = (-c3 * Sp * sign(x2 - x3) * sqrt(2 * g * abs(x2 - x3)) - c2 * Sp * sqrt(2 * g * x2) - qf + u2)/(A)
    dx[3] = (c1 * Sp * sign(x1 - x3) * sqrt(2 * g * abs(x1 - x3)) - c3 * Sp * sign(x3 - x2) * sqrt(2 * g * abs(x3 - x2)))/(A)
end
function ToyProblemF2!(dx, x, p, t)
    u1, u2, c1, Sp, g, A, c2, c3, alpha, r = p.u1, p.u2, p.c1, p.Sp, p.g, p.A, p.c2, p.c3, p.alpha, p.r
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    qf = c2 * pi * r ^ 2 * sqrt(2 * g * abs(x2))
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

#external variables have also been included in the parameters (must still be fixed)

function sample_params()
    return Params(
        5e-5,                            #u1
        5e-5,                            #u2
        rand(Normal(1.0, sqrt(0.0025))),       #c1
        5e-5,                            #m^2
        9.81,                            #g
        0.0154,                          #A
        rand(Normal(0.8, sqrt(0.0025))),       #c2
        rand(Normal(1.0, sqrt(0.0025))),       #c3
        rand(Normal(0.6, sqrt(4e-4))),         #alpha
        rand(Normal(0.002, sqrt(10e-9)))       #r should be 10e-6, but this introduces too much variance, it should be more like 10e-9.
    )
end

normalparams = Params(6e-5, 6e-5, 1, 5e-5, 9.81, 0.0154, 0.8, 1.0, 0.6, 0.002)

function ProcessWithoutSampling(faultmode, params)
    prob = ODEProblem(faultmode, x0, tspan, params)
    sol = solve(prob, Tsit5())
    return sol
end

function monte_carlo_simulation(N::Int, faultmode)
    x0 = [0.0, 0.0, 0.0]
    tspan = (0.0, 3000)
    measurements = zeros(N)

    for i in 1:N
        p = sample_params()
        prob = ODEProblem(faultmode, x0, tspan, p)
        sol = solve(prob, Tsit5(), saveat=tspan[2]) #only save final timestep
        measurements[i] = sol.u[end][3]
    end
    return measurements
end

function fitPDF(samples)
    hist = fit(Histogram, samples, nbins=50, closed=:left)
    bin_centers = (hist.edges[1][1:end-1] .+ hist.edges[1][2:end]) ./ 2
    pdf_estimate = hist.weights ./ (sum(hist.weights) * step(hist.edges[1]))
    return (bin_centers = bin_centers, pdf = pdf_estimate)
end

x0 = [0.01, 0.01, 0.01]
tspan = (0.0, 3000)

normal_sol = ProcessWithoutSampling(ToyProblemNormal!, normalparams)
fault1_sol = ProcessWithoutSampling(ToyProblemF1!, normalparams)
fault2_sol = ProcessWithoutSampling(ToyProblemF2!, normalparams)

#Comparison of plots

normal_y = getindex.(normal_sol.u, 3)
fault1_y = getindex.(fault1_sol.u, 3)
fault2_y = getindex.(fault2_sol.u, 3)

plot(normal_sol.t, normal_y, lab = "Normal")
plot!(fault1_sol.t, fault1_y, lab = "Fault 1")
plot!(fault2_sol.t, fault2_y, lab = "Fault 2")


#Monte Carlo monte_carlo_simulation

N = 1000

resultsNormal = monte_carlo_simulation(N, ToyProblemNormal!)
resultsF1 = monte_carlo_simulation(N, ToyProblemF1!)
resultsF2 = monte_carlo_simulation(N, ToyProblemF2!)

pdf_Normal = fitPDF(resultsNormal)
pdf_F1 = fitPDF(resultsF1)
pdf_F2 = fitPDF(resultsF2)

#In the actual toy problem this won't be the how the PDFs are generated, they will be the Guassians calculated from the update of probability. 

plot(pdf_Normal.bin_centers, pdf_Normal.pdf, label="Normal", xlabel="x", ylabel="Density", lw=2)
plot!(pdf_F1.bin_centers, pdf_F1.pdf, label="F1", xlabel="x", ylabel="Density", lw=2)
plot!(pdf_F2.bin_centers, pdf_F2.pdf, label="F2", xlabel="x", ylabel="Density", lw=2)

