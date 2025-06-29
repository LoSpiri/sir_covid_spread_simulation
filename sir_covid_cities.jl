using Agents, Random
using Agents.DataFrames, Agents.Graphs
using StatsBase: sample, Weights
using DrWatson: @dict
using LinearAlgebra: diagind
using IterTools
using Dates
using Distributions

@agent struct PoorSoul(GraphAgent)
    days_infected::Int  # number of days since is infected
    status::Symbol  # 1: S, 2: I, 3:R
end

function create_params(;
    C=4,
    max_travel_rate=0.01,
    lockdown_threshold=0.05,
    travel_reduction=0.9,
    infection_period=30,
    reinfection_probability=0.05,
    detection_time=14,
    death_rate=0.044,
    health_quality = rand(0.5:0.1:1.0, C),
    cities_population = rand(50:5000, C),
    urban_density = rand(0.2:0.1:0.8, C),
    cities_infected=[zeros(Int, C - 1)..., 1],
    seed=19,
)
    Random.seed!(seed)
    β_und = fill(1, C) .* urban_density
    β_det = β_und ./ 10
    cities_death_rate = death_rate .* (1 .- health_quality)

    Random.seed!(seed)
    migration_rates = zeros(C, C)
    for c in 1:C
        for c2 in 1:C
            migration_rates[c, c2] = (cities_population[c] + cities_population[c2]) / cities_population[c]
        end
    end
    maxM = maximum(migration_rates)
    migration_rates = (migration_rates .* max_travel_rate) ./ maxM
    migration_rates[diagind(migration_rates)] .= 1.0

    params = @dict(
        cities_population,
        β_und,
        β_det,
        migration_rates,
        infection_period,
        reinfection_probability,
        detection_time,
        cities_death_rate,
        cities_infected,
        lockdown_threshold,
        travel_reduction,
        seed
    )

    return params
end

function model_initiation(;
    cities_population,                                     # Popolazione per città
    migration_rates,                        # Tassi di migrazione tra città
    β_und, β_det,                           # Tassi di trasmissione (non rilevati/rilevati)
    lockdown_threshold,  # soglia per attivare lockdown
    travel_reduction,     # riduzione delle migrazioni durante lockdown
    infection_period,                    # Durata dell'infezione
    reinfection_probability,           # Probabilità di reinfezione
    detection_time,                      # Tempo per rilevamento
    cities_death_rate,                        # Tasso di mortalità
    cities_infected,  # Infetti iniziali
    seed,                                 # Seed per randomicità
)
    rng = Xoshiro(seed)
    @assert length(cities_population) ==
            length(cities_infected) ==
            length(β_und) ==
            length(β_det) ==
            size(migration_rates, 1) "length of cities_population, cities_infected, and B, and number of rows/columns in migration_rates should be the same "
    @assert size(migration_rates, 1) == size(migration_rates, 2) "migration_rates rates should be a square matrix"

    C = length(cities_population)
    # normalize migration_rates
    migration_rates_sum = sum(migration_rates, dims=2)
    for c in 1:C
        migration_rates[c, :] ./= migration_rates_sum[c]
    end

    properties = @dict(
        cities_population,
        cities_infected,
        β_und,
        β_det,
        β_det,
        migration_rates,
        infection_period,
        reinfection_probability,
        detection_time,
        C,
        cities_death_rate,
        lockdown_threshold,
        travel_reduction,
    )
    space = GraphSpace(complete_graph(C))
    model = StandardABM(PoorSoul, space; agent_step!, properties, rng)

    for city in 1:C, n in 1:cities_population[city]
        ind = add_agent!(city, model, 0, :S)
    end

    for city in 1:C
        inds = ids_in_position(city, model)
        for n in 1:cities_infected[city]
            agent = model[inds[n]]
            agent.status = :I
            agent.days_infected = 1
        end
    end
    return model
end

function agent_step!(agent, model)
    if agent.status == :D
        return
    end
    migrate!(agent, model)
    transmit!(agent, model)
    update!(agent, model)
    recover_or_die!(agent, model)
end

function migrate!(agent, model)
    pid = agent.pos
    
    infected_ratio = count(a.status == :I for a in allagents(model) if a.pos == pid) / model.cities_population[pid]
    if infected_ratio > model.lockdown_threshold
        move_prob = rand(abmrng(model))
        if move_prob < model.travel_reduction
            return
        end
    end

    m = sample(abmrng(model), 1:(model.C), Weights(model.migration_rates[pid, :]))
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

function transmit!(agent, model)
    agent.status ≠ :I && return
    transmission_rate = if agent.days_infected < model.detection_time
        model.β_und[agent.pos]
    else
        model.β_det[agent.pos]
    end

    base_people_met = rand(abmrng(model), Poisson(10))
    people_met = min(base_people_met, 100)
    pid = agent.pos
    infected_ratio = count(a.status == :I for a in allagents(model) if a.pos == pid) / model.cities_population[pid]
    if infected_ratio > model.lockdown_threshold
        people_met = max(1, round(Int, people_met * (1 - model.travel_reduction)))
    end

    contacts = 0
    potential_contacts = ids_in_position(agent, model)
    shuffle!(abmrng(model), potential_contacts)
    for contact_id in potential_contacts
        contacts ≥ people_met && break
        contact = model[contact_id]
        if contact.status == :S || 
           (contact.status == :R && rand(abmrng(model)) ≤ model.reinfection_probability)
            if rand(abmrng(model)) < transmission_rate
                contact.status = :I
                contact.days_infected = 0
            end
        end
        contacts += 1
    end
end

function update!(agent, model)
    if agent.status == :I
        agent.days_infected += 1
    end
end

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(abmrng(model)) ≤ model.cities_death_rate[agent.pos]
            agent.status = :D
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

function run_model_simulation(params, model, steps)
    @info "Starting simulation for $steps steps"
    
    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)
    dead(x) = count(i == :D for i in x)
    to_collect = [(:status, f) for f in (infected, recovered, dead, length)]

    @info "Collecting agent data..."
    data, _ = run!(model, steps; adata=to_collect)
    
    @debug "Simulation completed successfully"
    return data
end

function create_static_plots(model, data)
    @info "Creating static epidemic curve plot..."
    N = sum(model.cities_population)
    
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], xlabel="Steps", ylabel="log10(count)")
    
    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)
    dead(x) = count(i == :D for i in x)
    li = lines!(ax, data.time, log10.(data[:, dataname((:status, infected))]), color = :blue)
    lr = lines!(ax, data.time, log10.(data[:, dataname((:status, recovered))]), color = :red)
    ld = lines!(ax, data.time, log10.(data[:, dataname((:status, dead))]), color=:green)
    
    Legend(fig[1, 2], [li, lr, ld], ["infected", "recovered", "dead"])
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "static/plot/sir_covid_cities/epidemic_plot_$timestamp.png"
    save(filename, fig)
    @info "Saved epidemic curve plot to $(filename)"
end

function create_dynamic_visualization(params, steps)
    @info "Preparing dynamic visualization components..."
    model = model_initiation(; params...)
    
    abmobs = ABMObservable(model)
    
    infected_fraction(m, x) = count(m[id].status == :I for id in x) / length(x)
    infected_fractions(m) = [infected_fraction(m, ids_in_position(p, m)) for p in positions(m)]
    fracs = lift(infected_fractions, abmobs.model)
    color = lift(fs -> [cgrad(:inferno)[f] for f in fs], fracs)
    title = lift(
        (m) -> "Step = $(abmtime(m)), Infected = $(round(Int, 100*infected_fraction(m, allids(m))))%",
        abmobs.model
    )

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; title, xlabel="City", ylabel="Population")
    barplot!(ax, model.cities_population; strokecolor=:black, strokewidth=1, color)
    
    @info "Recording dynamic visualization ($(steps) steps)..."
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "static/video/sir_covid_cities/covid_evolution_$timestamp.mp4"
    record(fig, filename; framerate=2) do io
        for j in 1:steps
            @info "Recording frame $j/$(steps)"
            recordframe!(io)
            Agents.step!(abmobs, 1)
        end
        recordframe!(io)
    end
    @info "Saved dynamic visualization to $(filename)"
end

function create_grid_static_plots(models, datas, param_dicts_list)
    @info "Creating grid search comparison plots..."
    
    # Identify varying parameters
    varying_params = find_varying_parameters(param_dicts_list)
    @info "Varying parameters: $(keys(varying_params))"
    
    # Create comparison plots for each status
    create_comparison_plots(models, datas, param_dicts_list, varying_params)
    
    # Create parameter sensitivity analysis
    create_sensitivity_analysis(models, datas, param_dicts_list, varying_params)
end

function find_varying_parameters(param_dicts_list)
    if length(param_dicts_list) <= 1
        return Dict()
    end
    
    varying_params = Dict()
    
    # Get all parameter keys from first dict
    all_keys = keys(param_dicts_list[1])
    
    for key in all_keys
        # Get all values for this parameter across all combinations
        values = [params[key] for params in param_dicts_list]
        unique_values = unique(values)
        
        # If there's more than one unique value, it's varying
        if length(unique_values) > 1
            varying_params[key] = unique_values
        end
    end
    
    return varying_params
end

function create_comparison_plots(models, datas, param_dicts_list, varying_params)
    # Define colors for different parameter combinations
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]
    
    # Create plots for each status
    statuses = ["infected", "recovered", "dead"]
    
    for status in statuses
        fig = Figure(size=(800, 600))
        ax = Axis(fig[1, 1], 
                 xlabel="Steps", 
                 ylabel="log10(count)",
                 title="$(uppercasefirst(status)) - Parameter Comparison")
        
        legend_elements = []
        legend_labels = []
        
        for (i, (model, data, params)) in enumerate(zip(models, datas, param_dicts_list))
            N = sum(model.cities_population)
            color = colors[mod1(i, length(colors))]
            
            if status == "infected"
                infected(x) = count(i == :I for i in x)
                y_data = log10.(data[:, dataname((:status, infected))])
            elseif status == "recovered"
                recovered(x) = count(i == :R for i in x)
                y_data = log10.(data[:, dataname((:status, recovered))])
            else 
                dead(x) = count(i == :D for i in x)
                y_data = log10.(data[:, dataname((:status, dead))])
            end
            
            line = lines!(ax, data.time, y_data, color=color, linewidth=2)
            
            # Create label based on varying parameters
            label = create_param_label(params, varying_params)
            push!(legend_elements, line)
            push!(legend_labels, label)
        end
        
        Legend(fig[1, 2], legend_elements, legend_labels)
        
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = "static/plot/sir_covid_cities/grid_$(status)_comparison_$timestamp.png"
        save(filename, fig)
        @info "Saved $(status) comparison plot to $(filename)"
    end
end

function create_sensitivity_analysis(models, datas, param_dicts_list, varying_params)
    # Create multiple analysis plots
    fig = Figure(size=(1400, 800))
    
    n_combinations = length(param_dicts_list)
    max_infected = Float64[]
    max_recovered = Float64[]
    max_dead = Float64[]
    total_population = Float64[]
    labels = String[]
    
    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)
    dead(x) = count(i == :D for i in x)
    
    for (model, data, params) in zip(models, datas, param_dicts_list)
        N = sum(model.cities_population)
        
        # Get maximum values across all time steps
        push!(max_infected, maximum(data[:, dataname((:status, infected))]))
        push!(max_recovered, maximum(data[:, dataname((:status, recovered))]))
        push!(max_dead, maximum(data[:, dataname((:status, dead))]))
        push!(total_population, N)
        push!(labels, create_param_label(params, varying_params))
    end
    
    # Plot 1: Absolute counts with log scale
    ax1 = Axis(fig[1, 1], 
               xlabel="Parameter Combinations", 
               ylabel="Count (log scale)",
               title="Epidemic Outcomes - Absolute Counts",
               xticks=(1:n_combinations, labels),
               xticklabelrotation=π/4,
               yscale=log10)
    
    x_pos = 1:n_combinations
    width = 0.25
    
    # Add small offset to avoid log(0)
    safe_infected = max.(max_infected, 1)
    safe_recovered = max.(max_recovered, 1)
    safe_dead = max.(max_dead, 1)
    
    barplot!(ax1, x_pos .- width, safe_infected, width=width, color=:blue, label="Infected")
    barplot!(ax1, x_pos, safe_recovered, width=width, color=:red, label="Recovered") 
    barplot!(ax1, x_pos .+ width, safe_dead, width=width, color=:green, label="Dead")
    
    Legend(fig[1, 2], ax1)
    
    # Plot 2: Percentages
    ax2 = Axis(fig[2, 1], 
               xlabel="Parameter Combinations", 
               ylabel="Percentage (%)",
               title="Epidemic Outcomes - Percentages",
               xticks=(1:n_combinations, labels),
               xticklabelrotation=π/4)
    
    pct_infected = (max_infected ./ total_population) .* 100
    pct_recovered = (max_recovered ./ total_population) .* 100
    pct_dead = (max_dead ./ total_population) .* 100
    
    barplot!(ax2, x_pos .- width, pct_infected, width=width, color=:blue, label="Infected %")
    barplot!(ax2, x_pos, pct_recovered, width=width, color=:red, label="Recovered %") 
    barplot!(ax2, x_pos .+ width, pct_dead, width=width, color=:green, label="Dead %")
    
    Legend(fig[2, 2], ax2)
    
    # Plot 3: Relative differences from baseline (first combination)
    if n_combinations > 1
        ax3 = Axis(fig[1:2, 3], 
                   xlabel="Parameter Combinations", 
                   ylabel="Relative Change from Baseline (%)",
                   title="Relative Changes from First Combination (Peak Values)",
                   xticks=(2:n_combinations, labels[2:end]),
                   xticklabelrotation=π/4)
        
        # Calculate relative changes (excluding first combination as baseline)
        baseline_infected = max_infected[1]
        baseline_recovered = max_recovered[1]
        baseline_dead = max_dead[1]
        
        rel_infected = ((max_infected[2:end] .- baseline_infected) ./ baseline_infected) .* 100
        rel_recovered = ((max_recovered[2:end] .- baseline_recovered) ./ baseline_recovered) .* 100
        rel_dead = ((max_dead[2:end] .- baseline_dead) ./ baseline_dead) .* 100
        
        x_pos_rel = 2:n_combinations
        
        barplot!(ax3, x_pos_rel .- width, rel_infected, width=width, color=:blue, label="Infected")
        barplot!(ax3, x_pos_rel, rel_recovered, width=width, color=:red, label="Recovered") 
        barplot!(ax3, x_pos_rel .+ width, rel_dead, width=width, color=:green, label="Dead")
        
        # Add horizontal line at 0
        hlines!(ax3, [0], color=:black, linestyle=:dash, alpha=0.5)
        
        Legend(fig[1, 4], ax3)
    end
    
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filename = "static/plot/sir_covid_cities/grid_sensitivity_analysis_$timestamp.png"
    save(filename, fig)
    @info "Saved sensitivity analysis plot to $(filename)"
    
    # Also create a summary table with peak values
    create_sensitivity_summary_table(labels, max_infected, max_recovered, max_dead, total_population)
end

function create_sensitivity_summary_table(labels, infected_values, recovered_values, dead_values, total_population)
    @info "=== SENSITIVITY ANALYSIS SUMMARY ==="
    println("Parameter Combination | Infected | Recovered | Dead | Total Pop | Infected% | Recovered% | Dead%")
    println("-" ^ 90)
    
    for i in 1:length(labels)
        pct_infected = round((infected_values[i] / total_population[i]) * 100, digits=2)
        pct_recovered = round((recovered_values[i] / total_population[i]) * 100, digits=2)
        pct_dead = round((dead_values[i] / total_population[i]) * 100, digits=2)
        
        println("$(rpad(labels[i], 20)) | $(rpad(Int(infected_values[i]), 8)) | $(rpad(Int(recovered_values[i]), 9)) | $(rpad(Int(dead_values[i]), 4)) | $(rpad(Int(total_population[i]), 9)) | $(rpad(pct_infected, 9))% | $(rpad(pct_recovered, 10))% | $(rpad(pct_dead, 4))%")
    end
    
    if length(labels) > 1
        println("\n=== RELATIVE CHANGES FROM BASELINE ($(labels[1])) ===")
        baseline_infected = infected_values[1]
        baseline_recovered = recovered_values[1]
        baseline_dead = dead_values[1]
        
        for i in 2:length(labels)
            rel_infected = round(((infected_values[i] - baseline_infected) / baseline_infected) * 100, digits=2)
            rel_recovered = round(((recovered_values[i] - baseline_recovered) / baseline_recovered) * 100, digits=2)
            rel_dead = round(((dead_values[i] - baseline_dead) / baseline_dead) * 100, digits=2)
            
            println("$(rpad(labels[i], 20)) | Infected: $(rel_infected)% | Recovered: $(rel_recovered)% | Dead: $(rel_dead)%")
        end
    end
end

function create_param_label(params, varying_params)
    # Create a concise label showing only the varying parameter values
    label_parts = String[]
    
    for (param_name, _) in varying_params
        value = params[param_name]
        # Format the parameter name and value nicely
        param_str = string(param_name)
        if param_str == "lockdown_threshold"
            push!(label_parts, "LT=$(value)")
        elseif param_str == "travel_reduction"
            push!(label_parts, "TR=$(value)")
        else
            push!(label_parts, "$(param_str)=$(value)")
        end
    end
    
    return join(label_parts, ", ")
end

function run_simulation(grid=false, steps=10, num_cities=4)
    @info "===== EPIDEMIC SIMULATION STARTED ====="
    
    if (grid)
        grid_parameters = Dict(
            :C => [num_cities],
            :max_travel_rate => [0.01],
            :lockdown_threshold => [0.05, 0.2],
            :travel_reduction => [0.75, 0.9, 0.99],
            :infection_period => [14],
            :reinfection_probability => [0.1],
            :detection_time => [7],
            :death_rate => [0.044],
            :health_quality => [rand(0.8:0.1:1.0, num_cities)],
            :cities_population => [rand(50:5000, num_cities)],
            :urban_density => [rand(0.2:0.1:0.8, num_cities)],
            :cities_infected => [[zeros(Int, num_cities - 1)..., 1]],
            :seed => [19]
        )

        # Extract parameter names and values in consistent order
        param_names = collect(keys(grid_parameters))
        param_values = [grid_parameters[name] for name in param_names]

        # Generate all combinations using Iterators.product
        all_combinations = collect(Iterators.product(param_values...))

        # Create list of parameter dictionaries
        param_dicts_list = []

        for combination in all_combinations
            # Create keyword arguments dict for this combination
            kwargs = Dict()
            for (i, param_name) in enumerate(param_names)
                kwargs[param_name] = combination[i]
            end
            
            # Call create_params with the current combination of parameters
            param_dict = create_params(; kwargs...)
            
            # Add to our list
            push!(param_dicts_list, param_dict)
        end

        datas_list = []
        models_list = []
        for params in param_dicts_list
            # 2. Model Initialization
            @info "Initializing model..."
            model = model_initiation(; params...)
            # log_model_stats(model)
            
            # 3. Run Initial Simulation
            @info "Running initial simulation ($(steps) steps)..."
            data = run_model_simulation(params, model, steps)
            # log_simulation_results(data)
            push!(datas_list, data)
            push!(models_list, model)
        end

        # 4. Create Static Plots
        @info "Generating static visualizations..."
        create_grid_static_plots(models_list, datas_list, param_dicts_list)
    else
        # 1. Parameter Creation
        @info "Creating simulation parameters..."
        params = create_params()
        # log_parameters(params)
        
        # 2. Model Initialization
        @info "Initializing model..."
        model = model_initiation(; params...)
        # log_model_stats(model)
        
        # 3. Run Initial Simulation
        @info "Running initial simulation ($(steps) steps)..."
        data = run_model_simulation(params, model, steps)
        # log_simulation_results(data)
    
        # 4. Create Static Plots
        @info "Generating static visualizations..."
        create_static_plots(model, data)
        
        # 5. Create Dynamic Visualization
        @info "Preparing dynamic visualization..."
        create_dynamic_visualization(params, steps)
    end

    
    @info "===== SIMULATION COMPLETE ====="
end

using CairoMakie
run_simulation(true, 80, 3)