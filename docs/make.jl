using Documenter, FLOWMath

makedocs(;
    modules=[FLOWMath],
    format=Documenter.HTML(;
        repolink="https://github.com/byuflowlab/FLOWMath.jl/blob/{commit}{path}#L{line}",
        edit_link="master",
    ),
    pages=["Home" => "index.md"],
    sitename="FLOWMath.jl",
    authors="Andrew Ning <aning@byu.edu>",
    checkdocs=:exports,
)

deploydocs(; repo="github.com/byuflowlab/FLOWMath.jl.git")
