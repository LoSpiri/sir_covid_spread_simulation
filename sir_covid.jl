using Agents, Random
using Agents.DataFrames, Agents.Graphs
using StatsBase: sample, Weights
using DrWatson: @dict
using LinearAlgebra: diagind

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
    death_rate=0.04,
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
        if move_prob > model.travel_reduction
            return
        end
    end

    m = sample(abmrng(model), 1:(model.C), Weights(model.migration_rates[pid, :]))
    if m ≠ pid
        move_agent!(agent, m, model)
    end
end

function transmit!(agent, model)
    agent.status == :S && return
    rate = if agent.days_infected < model.detection_time
        model.β_und[agent.pos]
    else
        model.β_det[agent.pos]
    end

    n = rate * abs(randn(abmrng(model)))
    n <= 0 && return

    for contactID in ids_in_position(agent, model)
        contact = model[contactID]
        if contact.status == :S ||
           (contact.status == :R && rand(abmrng(model)) ≤ model.reinfection_probability)
            contact.status = :I
            n -= 1
            n <= 0 && return
        end
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
            remove_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

function run_model_simulation(model, steps)
    @info "Starting simulation for $steps steps"
    
    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)
    to_collect = [(:status, f) for f in (infected, recovered, length)]
    
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
    li = lines!(ax, data.time, log10.(data[:, dataname((:status, infected))]), color = :blue)
    lr = lines!(ax, data.time, log10.(data[:, dataname((:status, recovered))]), color = :red)
    dead = log10.(N .- data[:, dataname((:status, length))])
    ld = lines!(ax, data.time, dead, color=:green)
    
    Legend(fig[1, 2], [li, lr, ld], ["infected", "recovered", "dead"])
    save("epidemic_plot.png", fig)
    @info "Saved epidemic curve plot to epidemic_plot.png"
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
    record(fig, "covid_evolution.mp4"; framerate=2) do io
        for j in 1:steps
            @info "Recording frame $j/$(steps)"
            recordframe!(io)
            Agents.step!(abmobs, 1)
        end
        recordframe!(io)
    end
    @info "Saved dynamic visualization to covid_evolution.mp4"
end

function run_simulation()
    @info "===== EPIDEMIC SIMULATION STARTED ====="
    
    # 1. Parameter Creation
    @info "Creating simulation parameters..."
    params = create_params()
    # log_parameters(params)
    
    # 2. Model Initialization
    @info "Initializing model..."
    model = model_initiation(; params...)
    # log_model_stats(model)
    
    # 3. Run Initial Simulation
    steps = 10
    @info "Running initial simulation ($(steps) steps)..."
    data = run_model_simulation(model, steps)
    # log_simulation_results(data)

    # 4. Create Static Plots
    @info "Generating static visualizations..."
    create_static_plots(model, data)
    
    # 5. Create Dynamic Visualization
    @info "Preparing dynamic visualization..."
    create_dynamic_visualization(params, steps)
    
    @info "===== SIMULATION COMPLETE ====="
end

using CairoMakie
run_simulation()