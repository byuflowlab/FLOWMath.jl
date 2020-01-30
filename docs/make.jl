using Documenter, FLOWMath

makedocs(;
    modules=[FLOWMath],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/byuflowlab/FLOWMath.jl/blob/{commit}{path}#L{line}",
    sitename="FLOWMath.jl",
    authors="Andrew Ning <aning@byu.edu>",
    # assets=String[],
)

deploydocs(;
    repo="github.com/byuflowlab/FLOWMath.jl.git",
)
