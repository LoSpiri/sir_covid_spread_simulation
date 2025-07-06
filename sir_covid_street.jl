using Agents
using Random
using CairoMakie, OSMMakie
using DataFrames
using Statistics

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

const PISA_BOUNDS = (10.390, 43.710, 10.410, 43.725)
const PIAZZA_GARIBALDI = (10.4017, 43.7181)
const LEANING_TOWER_AREA = (10.3947, 43.7230)
const MAP_FILE = "pisa_map.osm"
const ROUTE_LIMIT = 50
const STATIONARY_MOVEMENT_PROBABILITY = 0.1
const TRANSMISSION_RADIUS = 0.01  # 10 meters

# Comparative analysis parameters
const SIMULATION_STEPS = 300
const REPLICATIONS = 4

# =============================================================================
# AGENT DEFINITION
# =============================================================================

@agent struct Person(OSMAgent)
    status::Symbol  # :susceptible, :infected, :recovered
    speed::Float64
    infection_time::Int  # Days since infection
end

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

"""
Download and setup Pisa map data if not already present.
"""
function setup_map()
    if !isfile(MAP_FILE)
        OSM.download_osm_network(
            :bbox;
            minlat = PISA_BOUNDS[2],
            minlon = PISA_BOUNDS[1], 
            maxlat = PISA_BOUNDS[4],
            maxlon = PISA_BOUNDS[3],
            download_format = :osm,
            save_to_file_location = MAP_FILE
        )
    end
    return MAP_FILE
end

"""
Create model properties dictionary.
"""
function create_model_properties(transmission_rate::Float64)
    return Dict(
        :dt => 1 / 60,
        :recovery_period => 14,  # Days to recover
        :transmission_rate => transmission_rate
    )
end

"""
Create and add susceptible population to the model.
"""
function add_susceptible_population!(model, population_size::Int)
    for id in 1:population_size
        start = random_position(model)
        speed = rand(abmrng(model)) * 5.0 + 2.0
        person = add_agent!(start, Person, model, :susceptible, speed, 0)
        OSM.plan_random_route!(person, model; limit = ROUTE_LIMIT)
    end
end

"""
Create and add patient zero to the model.
"""
function add_patient_zero!(model)
    start = OSM.nearest_road(PIAZZA_GARIBALDI, model)
    finish = OSM.nearest_node(LEANING_TOWER_AREA, model)
    speed = rand(abmrng(model)) * 5.0 + 2.0
    patient_zero = add_agent!(start, model, :infected, speed, 0)
    plan_route!(patient_zero, finish, model)
end

"""
Initialize the complete population model.
"""
function initialise_population(population_size::Int, transmission_rate::Float64; seed = 16)
    map_path = setup_map()
    properties = create_model_properties(transmission_rate)
    
    model = StandardABM(
        Person,
        OpenStreetMapSpace(map_path);
        agent_step! = person_step!,
        properties = properties,
        rng = Random.MersenneTwister(seed)
    )

    add_susceptible_population!(model, population_size)
    add_patient_zero!(model)
    
    return model
end

# =============================================================================
# AGENT MOVEMENT LOGIC
# =============================================================================

"""
Handle agent movement along routes.
"""
function handle_movement!(agent, model)
    distance_left = move_along_route!(agent, model, agent.speed * model.dt)

    if is_stationary(agent, model) && rand(abmrng(model)) < STATIONARY_MOVEMENT_PROBABILITY
        OSM.plan_random_route!(agent, model; limit = ROUTE_LIMIT)
        move_along_route!(agent, model, distance_left)
    end
end

# =============================================================================
# DISEASE PROGRESSION LOGIC
# =============================================================================

"""
Update infection time for infected agents.
"""
function update_infection_time!(agent)
    if agent.status == :infected
        agent.infection_time += 1
    end
end

"""
Check if agent should recover based on infection period.
"""
function check_recovery!(agent, model)
    if agent.status == :infected && agent.infection_time >= model.recovery_period * 24 * 60
        agent.status = :recovered
    end
end

"""
Handle disease transmission to nearby agents.
"""
function handle_transmission!(agent, model)
    if agent.status == :infected
        nearby_agents = nearby_ids(agent, model, TRANSMISSION_RADIUS)
        for nearby_id in nearby_agents
            nearby_person = model[nearby_id]
            if nearby_person.status == :susceptible && rand(abmrng(model)) < model.transmission_rate
                nearby_person.status = :infected
                nearby_person.infection_time = 0
            end
        end
    end
end

"""
Execute complete agent step with movement and disease progression.
"""
function person_step!(agent, model)
    handle_movement!(agent, model)
    update_infection_time!(agent)
    check_recovery!(agent, model)
    handle_transmission!(agent, model)
end

# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

"""
Count agents by status.
"""
function count_status(model)
    susceptible = count(agent -> agent.status == :susceptible, allagents(model))
    infected = count(agent -> agent.status == :infected, allagents(model))
    recovered = count(agent -> agent.status == :recovered, allagents(model))
    return (susceptible, infected, recovered)
end

"""
Collect SIR data for a single simulation run.
"""
function collect_sir_data(model, steps::Int)
    # Pre-allocate vectors for better performance
    step_data = Int[]
    susceptible_data = Int[]
    infected_data = Int[]
    recovered_data = Int[]
    
    for step in 0:steps
        s, i, r = count_status(model)
        push!(step_data, step)
        push!(susceptible_data, s)
        push!(infected_data, i)
        push!(recovered_data, r)
        step!(model)
    end
    
    sir_data = DataFrame(
        step = step_data,
        susceptible = susceptible_data,
        infected = infected_data,
        recovered = recovered_data
    )
    
    return sir_data
end

# =============================================================================
# COMPARATIVE ANALYSIS FUNCTIONS
# =============================================================================

"""
Run multiple simulations for parameter combinations.
"""
function run_comparative_analysis()
    # Parameter combinations
    population_sizes = [200, 800]  # Low and high population
    transmission_rates = [0.05, 0.2]  # Low and high transmission rates
    
    # Labels for scenarios
    scenario_labels = [
        "Low Pop, Low Trans",
        "Low Pop, High Trans", 
        "High Pop, Low Trans",
        "High Pop, High Trans"
    ]
    
    all_results = Dict()
    
    println("Running comparative analysis...")
    
    scenario_idx = 1
    for pop_size in population_sizes
        for trans_rate in transmission_rates
            println("Running scenario: $(scenario_labels[scenario_idx])")
            println("  Population: $pop_size, Transmission Rate: $trans_rate")
            
            scenario_results = []
            
            # Run multiple replications
            for rep in 1:REPLICATIONS
                println("  Replication $rep/$REPLICATIONS")
                
                # Create model with specific parameters
                model = initialise_population(pop_size, trans_rate; seed = rep * 42)
                
                # Collect data
                sir_data = collect_sir_data(model, SIMULATION_STEPS)
                
                # Add scenario information as separate columns
                sir_data[!, :scenario] .= scenario_labels[scenario_idx]
                sir_data[!, :replication] .= rep
                sir_data[!, :population_size] .= pop_size
                sir_data[!, :transmission_rate] .= trans_rate
                
                push!(scenario_results, sir_data)
            end
            
            all_results[scenario_labels[scenario_idx]] = scenario_results
            scenario_idx += 1
        end
    end
    
    return all_results
end

"""
Calculate summary statistics across replications.
"""
function calculate_summary_stats(results)
    summary_data = DataFrame()
    
    for (scenario, scenario_results) in results
        # Combine all replications for this scenario
        combined_data = vcat(scenario_results...)
        
        # Calculate mean and standard deviation for each time step
        grouped = groupby(combined_data, :step)
        scenario_summary = combine(grouped) do df
            DataFrame(
                scenario = df.scenario[1],
                population_size = df.population_size[1],
                transmission_rate = df.transmission_rate[1],
                susceptible_mean = mean(df.susceptible),
                susceptible_std = std(df.susceptible),
                infected_mean = mean(df.infected),
                infected_std = std(df.infected),
                recovered_mean = mean(df.recovered),
                recovered_std = std(df.recovered)
            )
        end
        
        append!(summary_data, scenario_summary)
    end
    
    return summary_data
end

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

"""
Determine agent color based on status.
"""
function person_color(agent)
    if agent.status == :susceptible
        return :blue
    elseif agent.status == :infected
        return :red
    else  # recovered
        return :green
    end
end

"""
Determine agent size based on status.
"""
function person_size(agent)
    if agent.status == :infected
        return 10
    else
        return 8
    end
end

"""
Create comparative plots showing SIR dynamics.
"""
function create_comparative_plots(summary_data)
    # Create figure with subplots
    fig = Figure(size = (1200, 800))
    
    # Define colors for scenarios
    colors = [:blue, :red, :green, :orange]
    scenarios = unique(summary_data.scenario)
    
    # Plot 1: Infected population over time
    ax2 = Axis(fig[1, 1], 
              title = "Infected Population Over Time",
              xlabel = "Time Steps",
              ylabel = "Number of Infected Individuals")
    
    for (i, scenario) in enumerate(scenarios)
        scenario_data = filter(row -> row.scenario == scenario, summary_data)
        lines!(ax2, scenario_data.step, scenario_data.infected_mean, 
               color = colors[i], linewidth = 2, label = scenario)
        band!(ax2, scenario_data.step, 
              scenario_data.infected_mean .- scenario_data.infected_std,
              scenario_data.infected_mean .+ scenario_data.infected_std,
              color = (colors[i], 0.2))
    end
    
    axislegend(ax2, position = :lt)
    
    # Plot 4: Peak infection comparison
    ax4 = Axis(fig[1, 2], 
              title = "Peak Infection Comparison",
              xlabel = "Scenario",
              ylabel = "Peak Number of Infected")
    
    peak_infections = []
    scenario_names = []
    
    for scenario in scenarios
        scenario_data = filter(row -> row.scenario == scenario, summary_data)
        peak_infection = maximum(scenario_data.infected_mean)
        push!(peak_infections, peak_infection)
        push!(scenario_names, scenario)
    end
    
    barplot!(ax4, 1:length(scenarios), peak_infections, 
             color = colors[1:length(scenarios)],
             strokecolor = :black, strokewidth = 1)
    
    ax4.xticks = (1:length(scenarios), scenario_names)
    ax4.xticklabelrotation = π/4
    
    # Add overall title
    Label(fig[0, :], "COVID-19 Outbreak Simulation: Comparative Analysis", 
          fontsize = 20, font = :bold)
    
    return fig
end

"""
Print summary statistics.
"""
function print_summary_statistics(summary_data)
    println("\n" * "="^80)
    println("SUMMARY STATISTICS")
    println("="^80)
    
    scenarios = unique(summary_data.scenario)
    
    for scenario in scenarios
        scenario_data = filter(row -> row.scenario == scenario, summary_data)
        peak_infected = maximum(scenario_data.infected_mean)
        peak_time = scenario_data.step[argmax(scenario_data.infected_mean)]
        final_recovered = scenario_data.recovered_mean[end]
        
        println("\n$scenario:")
        println("  Peak Infections: $(round(Int, peak_infected))")
        println("  Peak Time: $peak_time steps")
        println("  Final Recovered: $(round(Int, final_recovered))")
        println("  Attack Rate: $(round(final_recovered / scenario_data.population_size[1] * 100, digits=1))%")
    end
    
    println("\n" * "="^80)
end

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

"""
Run the original simulation with video output.
"""
function run_original_simulation()
    println("Running original simulation with video output...")
    covid_model = initialise_population(500, 0.1)
    
    abmvideo("static/video/sir_covid_street/pisa_covid_outbreak.mp4", covid_model;
        title = "COVID-19 Outbreak in Pisa, Italy (SIR Model)", 
        framerate = 15, 
        frames = 600,
        agent_color = person_color, 
        agent_size = person_size
    )
    
    println("Video saved as 'pisa_covid_outbreak.mp4'")
end

"""
Run the complete comparative analysis.
"""
function run_complete_analysis()
    println("Starting comprehensive COVID-19 simulation analysis...")
    
    # Run comparative analysis
    results = run_comparative_analysis()
    
    # Calculate summary statistics
    summary_data = calculate_summary_stats(results)
    
    # Create and save comparative plots
    fig = create_comparative_plots(summary_data)
    save("static/plot/sir_covid_street/pisa_covid_comparative_analysis.png", fig)
    println("Comparative analysis plot saved as 'pisa_covid_comparative_analysis.png'")
    
    # Print summary statistics
    print_summary_statistics(summary_data)
    
    # Display the plot
    display(fig)
    
    return results, summary_data
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Run both the original simulation and comparative analysis
println("COVID-19 Simulation in Pisa - Extended Analysis")
println("="^50)

# Uncomment the line below to run the original simulation with video
# run_original_simulation()

# Run the comparative analysis
results, summary_data = run_complete_analysis()

println("\nAnalysis complete! Check the generated plots and summary statistics.")