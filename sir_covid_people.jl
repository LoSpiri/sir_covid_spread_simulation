using Agents, Random
using Dates
using CairoMakie
using DrWatson: @dict

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

const STEPS_PER_DAY = 24
const DEFAULT_SIMULATION_STEPS = 3000
const DEFAULT_VIDEO_FRAMES = 300

# =============================================================================
# AGENT DEFINITION
# =============================================================================

@agent struct PoorSoul(ContinuousAgent{2, Float64})
    mass::Float64
    days_infected::Int
    status::Symbol
    β::Float64
end

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

"""
Initialize SIR epidemiological model with specified parameters.
Returns a StandardABM model ready for simulation.
"""
function sir_initiation(;
    infection_period = 14 * STEPS_PER_DAY,
    detection_time = 7 * STEPS_PER_DAY,
    reinfection_probability = 0.1,
    isolated = 0.0,
    interaction_radius = 0.012,
    dt = 1.0,
    speed = 0.002,
    death_rate = 0.044, # from website of WHO
    N = 1000,
    initial_infected = 5,
    seed = 42,
    βmin = 0.4,
    βmax = 0.8,
)
    # Validate parameters
    @assert 0 ≤ isolated ≤ 1 "Isolation rate must be between 0 and 1"
    @assert initial_infected ≤ N "Initial infected cannot exceed total population"
    @assert βmin ≤ βmax "βmin must be ≤ βmax"
    
    properties = (;
        infection_period,
        reinfection_probability,
        detection_time,
        death_rate,
        interaction_radius,
        dt,
    )
    
    space = ContinuousSpace((1,1); spacing = 0.02)
    model = StandardABM(PoorSoul, space, agent_step! = sir_agent_step!,
                        model_step! = sir_model_step!, properties = properties,
                        rng = MersenneTwister(seed))

    create_population!(model, N, initial_infected, isolated, speed, βmin, βmax)
    return model
end

"""
Create the initial population of agents with specified characteristics.
"""
function create_population!(model, N, initial_infected, isolated, speed, βmin, βmax)
    for ind in 1:N
        pos = Tuple(rand(abmrng(model), 2))
        status = ind ≤ N - initial_infected ? :S : :I
        isisolated = ind ≤ isolated * N
        mass = isisolated ? Inf : 1.0
        vel = isisolated ? (0.0, 0.0) : sincos(2π * rand(abmrng(model))) .* speed

        β = (βmax - βmin) * rand(abmrng(model)) + βmin
        add_agent!(pos, model, vel, mass, 0, status, β)
    end
end

# =============================================================================
# TRANSMISSION AND INTERACTION MECHANICS
# =============================================================================

"""
Handle disease transmission between two agents.
"""
function transmit!(model, a1, a2, rp)
    # Only one agent must be infected for transmission to occur
    count(a.status == :I for a in (a1, a2)) ≠ 1 && return
    infected, healthy = a1.status == :I ? (a1, a2) : (a2, a1)

    # Check if transmission occurs based on infected agent's β
    rand(abmrng(model)) > infected.β && return

    # Handle reinfection for recovered agents
    if healthy.status == :R
        rand(abmrng(model)) > rp && return
    end
    
    healthy.status = :I
end

"""
Execute model-level step: handle interactions and collisions.
"""
function sir_model_step!(model)
    r = model.interaction_radius
    
    for (a1, a2) in interacting_pairs(model, r, :nearest)
        transmit!(model, a1, a2, model.reinfection_probability)
        elastic_collision!(a1, a2, :mass)
    end
end

# =============================================================================
# AGENT BEHAVIOR AND DISEASE PROGRESSION
# =============================================================================

"""
Execute agent-level step: movement and disease progression.
"""
function sir_agent_step!(agent, model)
    if agent.status == :D
        return
    end
    move_agent!(agent, model, model.dt)
    update_infection_status!(agent)
    recover_or_die!(agent, model)
end

"""
Update agent's infection status and days infected.
"""
function update_infection_status!(agent)
    agent.status == :I && (agent.days_infected += 1)
end

"""
Handle recovery or death for infected agents.
"""
function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(abmrng(model)) ≤ model.death_rate
            agent.status = :D
            agent.mass = Inf
            agent.vel = (0.0, 0.0)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

# =============================================================================
# DATA COLLECTION AND ANALYSIS
# =============================================================================

"""
Color mapping for agent visualization based on status.
"""
sir_colors(a) = a.status == :S ? "#2176ff" :    # blue
               a.status == :I ? "#bf2642" :     # red
               a.status == :D ? "#000000" :     # black
               "#338c54"                        # green for :R

"""
Count infected agents in the population.
"""
infected(x) = count(i == :I for i in x)

"""
Count recovered agents in the population.
"""
recovered(x) = count(i == :R for i in x)

# Agent data collection configuration
const ADATA = [(:status, infected), (:status, recovered)]

# =============================================================================
# SIMULATION EXECUTION AND ANALYSIS
# =============================================================================

"""
Run a single simulation scenario with specified parameters.
"""
function run_simulation_scenario(name; kwargs...)
    @info "Running simulation scenario: $name"
    model = sir_initiation(; kwargs...)
    data, _ = run!(model, DEFAULT_SIMULATION_STEPS; adata=ADATA)
    final_infected = data[end, dataname((:status, infected))]
    final_recovered = data[end, dataname((:status, recovered))]
    return data
end

"""
Create and save comparison plots for multiple simulation scenarios.
"""
function create_comparison_plots()
    # Define scenario parameters
    r1, r2 = 0.04, 0.33
    β1, β2 = 0.5, 0.1
    
    # Run different scenarios
    data1 = run_simulation_scenario("Low reinfection, High β"; reinfection_probability = r1, βmin = β1)
    println(data1[(end-9):end, :])
    data2 = run_simulation_scenario("High reinfection, High β"; reinfection_probability = r2, βmin = β1)
    println(data2[(end-9):end, :])
    data3 = run_simulation_scenario("Low reinfection, Low β"; reinfection_probability = r1, βmin = β2)
    println(data3[(end-9):end, :])
    
    # Display final data sample
    
    # Create visualization
    figure = Figure()
    ax = figure[1, 1] = Axis(figure; ylabel = "Infected")
    
    l1 = lines!(ax, data1[:, dataname((:status, infected))], color = :orange)
    l2 = lines!(ax, data2[:, dataname((:status, infected))], color = :blue)
    l3 = lines!(ax, data3[:, dataname((:status, infected))], color = :green)
    
    figure[1, 2][1,1] = Legend(figure, [l1, l2, l3], 
    [
    "reinfect rate=$r1,\ntransmit prob=$β1",
    "reinfect rate=$r2,\ntransmit prob=$β1",
    "reinfect rate=$r1,\ntransmit prob=$β2"
    ])    
    # Add social distancing scenario
    r4 = 0.04
    data4 = run_simulation_scenario("Social distancing"; reinfection_probability = r4, βmin = β1, isolated = 0.8)
    
    l4 = lines!(ax, data4[:, dataname((:status, infected))], color = :red)
    figure[1, 2][2,1] = Legend(
        figure,
        [l4],
        ["reinfect rate=$r4,\ntransmit prob=$β1,\nisolated = 0.8"]
    )
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "static/plot/sir_covid_people/epidemic_plot_$timestamp.png"
    save(filename, figure)
    @info "Plot saved as '$(filename)'"
    return figure
end

"""
Create and save simulation video with social distancing visualization.
"""
function create_simulation_video()
    sir_model = sir_initiation(isolated = 0.8)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "static/video/sir_covid_people/social_distancing_$timestamp.mp4"
    abmvideo(
        filename,
        sir_model;
        title = "Social Distancing",
        frames = DEFAULT_VIDEO_FRAMES,
        dt = 2,
        agent_color = sir_colors,
        framerate = 20,
    )
    @info "Video saved as '$(filename)'"
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

"""
Execute the complete simulation analysis.
"""
function main()
    @info "=== Starting SIR Epidemiological Simulation Analysis ==="
    
    try
        # Create comparison plots
        @info "=== Creating comparison plots"
        figure = create_comparison_plots()
        
        # Create simulation video
        @info "=== Creating simulation video"
        create_simulation_video()
        
        @info "=== Simulation analysis completed successfully ==="
    catch e
        @error "Simulation failed with error: $e"
        rethrow(e)
    end
end

# Execute main analysis
main()