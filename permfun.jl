using LinearAlgebra: norm
using Plots
using Random
using StatsBase: cor, mean, sample

"""
    permutefun!(x::Vector, y::Vector, rule::Function, score::Real; tol::Number = 1e-3, max_iter::Int = 10_000, max_search::Number = 100, verbose::Bool = true)

Permute y values (in-place) to ensure that x, y comply with a specific per-sample rule.

# Arguments
- `x::Vector`: The vector of x values
- `y::Vector`: The vector of y values
- `rule::Function`: Function taking two numbers and outputting a single value
- `score::Real`: The target score, i.e., the target value of `sum(rule.(x, y))`
- `tol::Real`: The tolerance. If the loss `abs(current_score - score)` is below this value, stop the algorithm.
- `max_iter::Int`: Maximum number of iterations. For large datasets, this may need to be increased.
- `max_search::Int`: The number of iterations to search for improvements. For large datasets, this may need to be increased.
- `verbose::Bool`: Whether to print debug information

# Examples
Basic x less-than y constraint
```julia-repl
julia> x = [1, 2, 3, 4, 5]
julia> y = [6, 5, 4, 3, 2]
julia> permutefun!(x, y, <)
julia> x, y
([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
```

Achieve a target correlation
```julia-repl
julia> ρ = 1.0
julia> x = [1, 2, 3, 4, 5]
julia> y = [6, 5, 4, 3, 2]
julia> xm, ym = mean(x), mean(y)
julia> permutefun!(x, y, (xi, yi) -> (xi - xm)*(yi - ym), ρ*norm(x .- xm)*norm(y .- ym))
julia> x, y

"""
function permutefun!(x::Vector, y::Vector, rule::Function, score::Real; max_iter::Int=10_000, max_search::Int=100, tol::Real=1e-3, change_tol::Real=1e-5, verbose::Bool=true)::PermutationFunctionStatus
    N = length(x)
    if N != length(y)
        throw(ArgumentError("x and y should be the same length!"))
    end

    # init algorithm status
    status = PermutationFunctionStatus(max_iter, max_search, tol, change_tol)

    # current objective value
    current_rule  = rule.(x, y)
    current_score = sum(current_rule)
    status.loss   = abs(score - current_score)

    while status.iter < status.max_iter
        # get random index
        i = sample(1:N)

        # compute change in score and find new loss with optimal j
        delta_score = rule.(x, y[i]) .+ rule.(x[i], y) .- current_rule .- current_rule[i]
        new_loss, j = findmin(abs.(score .- (current_score .+ delta_score)))

        # only change if loss improves
        if new_loss < status.loss
            # Found option! make change
            y[i], y[j] = y[j], y[i]
            current_rule[i], current_rule[j] = rule(x[i], y[i]), rule(x[j], y[j])
            current_score += delta_score[j]
            status.loss = new_loss
            status.loss_change = -delta_score[j]
            if verbose
                println("Iter $(status.iter) | $i ↔ $j | score $current_score | loss $(status.loss) | search $(status.search)")
            end
            status.search = 0
        else
            # increment search counter
            status.search += 1
        end

        # stopping conditions
        if get_condition(status) != UNFINISHED break end

        # increment iteration counter
        status.iter += 1
    end
    return status;
end

# for boolean rules, no score target is required
function permutefun!(x::Vector, y::Vector, rule::Function; max_iter::Int=10_000, max_search::Int=100, tol::Real=1e-3, change_tol::Real=1e-5, verbose::Bool=true)::PermutationFunctionStatus
    if !isa(rule(x[begin], y[begin]), Bool)
        throw(ArgumentError("With non-Boolean `rule`, please specify a target `score`"))
    end
    return permutefun!(x, y, rule, length(x); max_iter=max_iter, max_search=max_search, tol=tol, change_tol=change_tol, verbose=verbose)
end

# Not in-place
function permutefun(x::Vector, y::Vector, rule::Function, score::Real; max_iter::Int=10_000, max_search::Int=100, tol::Real=1e-3, change_tol::Real=1e-5, verbose::Bool=true)::Tuple{Vector, PermutationFunctionStatus}
    yc = copy(y)
    status = permutefun!(x, yc, rule, score; max_iter=max_iter, max_search=max_search, tol=tol, change_tol=change_tol, verbose=verbose);
    return (yc, status)
end

function permutefun(x::Vector, y::Vector, rule::Function; max_iter::Int=10_000, max_search::Int=100, tol::Real=1e-3, change_tol::Real=1e-5, verbose::Bool=true)::Tuple{Vector, PermutationFunctionStatus}
    yc = copy(y)
    status = permutefun!(x, yc, rule; max_iter=max_iter, max_search=max_search, tol=tol, change_tol=change_tol, verbose=verbose);
    return (yc, status)
end

# helper functions
mutable struct PermutationFunctionStatus
    iter::Int
    max_iter::Int
    search::Int
    max_search::Int
    loss::Real
    tol::Real
    loss_change::Real
    change_tol::Real
end

function Base.show(io::IO, status::PermutationFunctionStatus)
    println("{{ Permutation function status }}")
    cond = get_condition(status)
    println("Status: $cond")
    println(get_message(status, cond))
    println("Iter $(status.iter) | loss $(status.loss)")
end

function PermutationFunctionStatus(max_iter=10_000, max_search=100, tol=1e-3, change_tol=1e-5)
    PermutationFunctionStatus(0, max_iter, 0, max_search, Inf, tol, Inf, change_tol)
end

@enum StoppingCondition begin
    # successful statuses
    LOSS_BELOW_TOL = 1
    CHANGE_BELOW_TOL = 2
    # failure statuses
    SEARCH_ABOVE_LIM = 3
    ITER_ABOVE_LIM = 4
    # unfinished status
    UNFINISHED = 5
end

function get_condition(a::PermutationFunctionStatus)::StoppingCondition
    a.loss < a.tol && return LOSS_BELOW_TOL
    abs(a.loss_change) < a.change_tol && return CHANGE_BELOW_TOL
    a.search >= a.max_search && return SEARCH_ABOVE_LIM
    a.iter >= a.max_iter && return ITER_ABOVE_LIM
    return UNFINISHED
end

function get_message(a::PermutationFunctionStatus, c::StoppingCondition)::String
    c == LOSS_BELOW_TOL   && return "Success. Loss ($(a.loss)) below the tolerance"
    c == CHANGE_BELOW_TOL && return "Success. Change $(a.loss_change) below the tolerance"
    c == SEARCH_ABOVE_LIM && return "Failure. No improvement found after $(a.iter - a.max_search) iterations"
    c == ITER_ABOVE_LIM   && return "Failure. Maximum iterations reached ($(a.iter))."
    return ""
end

# Try out and produce a plot
# Generate some data
N = 300
Random.seed!(45)
x = rand(N) .- 0.5
y = vcat(randn(Int(N / 2)) ./ 6 .- 0.25, randn(Int(N / 2)) ./ 8 .+ 0.35)
p1 = scatter(x, y, title="Original data", xlabel="x", ylabel="y")

# Enforce complex constraint
ynew, status = permutefun(x, y, (xi, yi) -> (xi^4 < yi^2))
p2 = scatter(x, ynew, title="Constraint x⁴ < y²", xlabel="x", ylabel="y")

# induce a certain correlation
xm, ym = mean(x), mean(y)
target = 0.7 * norm(x .- xm) * norm(y .- ym)
ynew, status = permutefun(x, y, (xi, yi) -> (xi - xm) * (yi - ym), target)
cor(x, ynew)
p3 = scatter(x, ynew, title="Correlation = .7", xlabel="x", ylabel="y")

# make a hole in the data
ynew, status = permutefun(x, y, (xi, yi) -> sqrt(xi^2 + yi^2) > 0.3)
p4 = scatter(x, ynew, title="Hole of radius 0.3", xlabel="x", ylabel="y")

# plot them all together
plot(p1, p2, p3, p4, size=(1600, 400), layout=(1, 4), margin=30Plots.px)
savefig("permutefun.pdf")

# animation
N = 800
Random.seed!(45)
x = rand(N) .- 0.5
y = vcat(randn(Int(N / 2)) ./ 6 .- 0.25, randn(Int(N / 2)) ./ 8 .+ 0.35)
anim = @animate for iter in 1:60
    status = permutefun!(x, y, (xi, yi) -> sqrt(xi^2 + yi^2) > 0.3; max_iter=10);
    scatter(x, y, show=true, reuse=true, size = (450, 400), xlim=(-.7, .7), ylim=(-.7, .7))
end

gif(anim, fps=15)
