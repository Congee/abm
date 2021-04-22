module Model

export hello

using LinearAlgebra
import Random
using Agents
using Distributions
import InteractiveDynamics # plotting agents
using AbstractPlotting # plotting data
import CairoMakie # for static plotting
using Statistics

const num_redditors = 100
const seed = 42
const space_side_length = 100

# Assumptions:
# * existing communities like r/pics, r/worldnews, r/politics, r/liberal, r/conservertive
# * agents are: newbie, evil source, paid propagandists, influenced propagandists
#

Pos = Tuple{Float64, Float64}  # Position in a cartesian coordinate system

mutable struct Redditor <: AbstractAgent
    id::Int
    pos::Pos
    vel::Pos

    initial_stance::Float64  # [-1, 1]; 0 means neutral
    stance::Float64  # [-1, 1]; 0 means neutral
    type::Symbol  # :regular, :paid
    initial_color::Symbol
end

mutable struct Post
    stance::Float64
    votes::Int
end

const rng = Random.MersenneTwister(seed)


"""
red if < 0.5
blue if > 0.5
"""
function create_model()::AgentBasedModel
    space = ContinuousSpace((space_side_length, space_side_length))
    params = Dict(
        :red_initial_stance => -0.6,
        :blue_initial_stance => 0.6,
        :gray_initial_stance => 0.0,

        :bandwagon_sensitivity => 0.2,
        :echo_chamber_coefficient => 0.1,
        :goebbels_sensitivity => 0.01,
    )
    properties = Dict(:posts => [], :tick => 0)


    model = AgentBasedModel(Redditor, space; rng, properties=merge(properties, params))
    return model
end


"""returns `num` coordinates closest to `pos` in a grid space"""
function rasterize(num::Int, pos::Pos)
    # store distance to pos from each cell in a grid
    vec = Vector{Tuple{Pos, Float64}}()
    for i = 0:space_side_length
        for j = 0:space_side_length
            euclid = abs(norm(pos .- (i, j)))
            push!(vec, ( (i, j), euclid ))
        end
    end

    sorted = sort!(vec, by=x->x[2])  # by distance
    collect(map(x -> x[1], Iterators.take(sorted, num)))
end


function initialize!(model::AgentBasedModel)::AgentBasedModel
    blue_ratio = 0.1
    red_ratio = 0.1
    blank_ratio = 0.8

    num_blue_agents = Int(num_redditors * blue_ratio)
    num_red_agents = Int(num_redditors * red_ratio)
    num_blank_agents = Int(num_redditors - num_blue_agents - num_red_agents)
    
    pos_blue_center = (25., 75.)
    pos_red_center = (75., 75.)
    pos_news_center = (50., 25.)

    blue_points = rasterize(num_blue_agents, pos_blue_center)
    red_points = rasterize(num_red_agents, pos_red_center)
    news_points = rasterize(num_blank_agents, pos_news_center)

    range = 1:num_blue_agents
    stances = make_stance(model.blue_initial_stance, 0.2, 1.0, length(range))
    for (id, pos, stance) in zip(range, blue_points, stances)
        agent = Redditor(id, pos, (0, 0), stance, stance, :regular, :red)
        add_agent_pos!(agent, model)
    end

    red_range = (num_blue_agents + 1):(num_red_agents + num_blue_agents)
    stances = make_stance(model.red_initial_stance, -1.0, 0.2, length(red_range))
    for (id, pos, stance) in zip(red_range, red_points, stances)
        agent = Redditor(id, pos, (0, 0), stance, stance, :regular, :blue)
        add_agent_pos!(agent, model)
    end

    news_range = (num_redditors - num_blank_agents + 1):num_redditors
    stances = make_stance(model.gray_initial_stance, -0.2, 0.2, length(news_range))
    for (id, pos, stance) in zip(news_range, news_points, stances)
        agent = Redditor(id, pos, (0, 0), stance, stance, :regular, :gray)
        add_agent_pos!(agent, model)
    end

    model
end


"""
What affect votes of an post:
* current votes - bandwagon effect unless 0
* post stance
* reader stance

"""
function vote(agent::Redditor, post::Post, model::AgentBasedModel)::Int
    # TODO: when agent is blank

    multiplier = 1 + (post.votes == 0 ? 0 : model.bandwagon_sensitivity)

    if sign(post.stance * agent.stance * multiplier) == 1  # same direction
        result = 1
    else
        diff_stance = abs(agent.stance - post.stance * multiplier)
        result = diff_stance > 0.2 ? -1 : 1
    end
    println("result: $(result), post.stance $(post.stance), agent.stance $(agent.stance)")
    return result
end


function draw!(model)
    # color
    blue = "#0000ff"
    red = "#ff0000"
    grey = "#d3d3d3"
    ac(agent::Redditor) = agent.stance < -0.5 ? blue : agent.stance > 0.5 ? red : grey


    """
    The bandwagon effect - dicision making of individuals is affected by trends.
    """
    function agent_step!(
        agent::Redditor,
        model::AgentBasedModel,
        # args
    )
        post = last(model.posts)
        println("agent_step! votes: $(post.votes), tick: $(model.tick)")

        if agent.initial_color == :gray
            if -1 < agent.stance < 1 || true
                agent.stance += post.votes > 0 ? post.stance * model.goebbels_sensitivity : 0
                # println(post.stance * model.goebbels_sensitivity)
            end

            v = vote(agent, post, model)  # -1 or 1
            println("v $(v)")
            post.votes += v

            if -1 < agent.stance < 1 || true
                agent.stance *= 1 + model.echo_chamber_coefficient * sign(post.stance)
                println("chamber $(1 + model.echo_chamber_coefficient * sign(post.stance))")
            end
        end

        model.tick += 1
    end

    model.posts = push!(model.posts, Post(0.6, 50))
    function model_step!(
        model::AgentBasedModel,
    )
        println("model_step! $(length(model.posts))")
        model.posts = push!(model.posts, Post(0.6, 50))
    end

    
    plotkwargs = (
        ac = ac,
    )
    # mdata_vote_fns = map(post -> model -> post.votes, model.posts)
    mdata_vote_fns(model) = model.posts[1].votes
    num_posts(model) = length(model.posts)
    # mlabel_votes = map(x -> "votes$(x[1])", enumerate(model.posts))
    # mlabel_votes(model) = "votes"  # model.posts[1].votes
    adata = [
             (:stance, mean, x -> x.initial_color == :gray),
             (:stance, mean, x -> x.initial_color == :gray && x.initial_stance > 0),
             (:stance, mean, x -> x.initial_color == :gray && x.initial_stance < 0),
            ]
    alabels = ["mean gray", "blue", "red"]

    params = (
        :red_initial_stance => -1.0:0.1:0.0,
        :blue_initial_stance => 0.0:0.1:1.0,
        :gray_initial_stance => -0.2:0.05:0.2,

        :bandwagon_sensitivity => 0.0:0.01:0.5,
        :echo_chamber_coefficient => 0.0:0.01:1.0,
        :goebbels_sensitivity => 0.0:0.01:0.5,
    )

    _ = InteractiveDynamics.abm_data_exploration(
        model, agent_step!, model_step!, params;
        # mdata = [num_posts],
        # mlabels = ["num posts"],
        adata = adata,
        alabels = alabels,
        plotkwargs...
    )
    return
end


"""helper function to allocate the initial stance of an agent"""
function make_stance(mean::Float64, lo::Float64, hi::Float64, size::Int)::Vector{Float64}
    mu = mean
    sigma = 1
    distribution = Truncated(Normal(mu, sigma), lo, hi)
    rand(rng, distribution, size)
end


function case1()
    model = create_model()
    model = initialize!(model)
    draw!(model)
end


hello() = case1()

end # module
