using Agents, Random
using Agents.DataFrames, Agents.Graphs
using StatsBase: sample, Weights
using DrWatson: @dict
using CairoMakie

@agent struct PoorSoul(GraphAgent)
    days_infected::Int  # number of days since is infected
    status::Symbol  # 1: S, 2: I, 3:R
end

function model_initiation(;
    cities_population,                                     # Popolazione per città
    migration_rates,                        # Tassi di migrazione tra città
    β_und, β_det,                           # Tassi di trasmissione (non rilevati/rilevati)
    lockdown_threshold=0.05,  # soglia per attivare lockdown
    travel_reduction=0.9,     # riduzione delle migrazioni durante lockdown
    infection_period=30,                    # Durata dell'infezione
    reinfection_probability=0.05,           # Probabilità di reinfezione
    detection_time=14,                      # Tempo per rilevamento
    death_rate=0.02,                        # Tasso di mortalità
    cities_infected=[zeros(Int, length(cities_population) - 1)..., 1],  # Infetti iniziali
    seed=0,                                 # Seed per randomicità
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
        infection_period,
        reinfection_probability,
        detection_time,
        C,
        death_rate,
        lockdown_threshold,
        travel_reduction
    )
    space = GraphSpace(complete_graph(C))
    model = StandardABM(PoorSoul, space; agent_step!, properties, rng)

    # Add initial individuals
    for city in 1:C, n in 1:cities_population[city]
        ind = add_agent!(city, model, 0, :S) # Susceptible
    end
    # add infected individuals
    for city in 1:C
        inds = ids_in_position(city, model)
        for n in 1:cities_infected[city]
            agent = model[inds[n]]
            agent.status = :I # Infected
            agent.days_infected = 1
        end
    end
    return model
end

using LinearAlgebra: diagind

function create_params(;
    C,
    max_travel_rate,
    lockdown_threshold=0.05,  # soglia per attivare lockdown
    travel_reduction=0.9,     # riduzione delle migrazioni durante lockdown
    infection_period=30,
    reinfection_probability=0.05,
    detection_time=14,
    death_rate=0.02,
    cities_infected=[zeros(Int, C - 1)..., 1],
    seed=19,
)

    Random.seed!(seed)
    cities_population = rand(50:5000, C)
    β_und = rand(0.3:0.02:0.6, C)
    β_det = β_und ./ 10

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
        death_rate,
        cities_infected,
        lockdown_threshold,
        travel_reduction
    )

    return params
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

update!(agent, model) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(abmrng(model)) ≤ model.death_rate
            remove_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

params = create_params(C=8, max_travel_rate=0.01)
model = model_initiation(; params...)



using CairoMakie
abmobs = ABMObservable(model)

infected_fraction(m, x) = count(m[id].status == :I for id in x) / length(x)
infected_fractions(m) = [infected_fraction(m, ids_in_position(p, m)) for p in positions(m)]
fracs = lift(infected_fractions, abmobs.model)
color = lift(fs -> [cgrad(:inferno)[f] for f in fs], fracs)
title = lift(
    (m) -> "step = $(abmtime(m)), infected = $(round(Int, 100*infected_fraction(m, allids(m))))%",
    abmobs.model
)

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1]; title, xlabel="City", ylabel="Population")
barplot!(ax, model.cities_population; strokecolor=:black, strokewidth=1, color)
fig

record(fig, "covid_evolution.mp4"; framerate=5) do io
    for j in 1:30
        recordframe!(io)
        Agents.step!(abmobs, 1)
    end
    recordframe!(io)
end