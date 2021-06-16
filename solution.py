import approximant
import series
import exact_critical_solution
import shifter
import repeater
import conservation_of_energy

def get_series_solution(conds, N):
    series_solution = repeater.repeat(conds, series.get_series_function(conds, N))
    series_deriv = repeater.repeat(conds, series.get_series_deriv_funct(conds, N))
    return shifter.shift(conds, series_solution, series_deriv)

def get_approximant(conds, N):
    behavior = conservation_of_energy.get_behavior(conds)
    if behavior in ('subcritical', 'supercritical'):
        approx = repeater.repeat(conds, approximant.get_approximant(conds, N))
        deriv = repeater.repeat(conds, approximant.get_approximant_derivative(conds, N))
    else:
        approx, deriv = exact_critical_solution.get_exact_critical_solution(conds), None
    return shifter.shift(conds, approx, deriv)

    
