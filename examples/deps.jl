import Libdl

const libbosque = joinpath(@__DIR__, "..", "target", "release", "libbosque.dylib")

function check_deps()
    global libbosque
    if !isfile(libbosque)
        error("$libbosque does not exist, Please re-run Pkg.build(\"Bosque\"), and restart Julia.")
    end

    if Libdl.dlopen_e(libbosque) == C_NULL
        error("$libbosque cannot be opened, Please re-run Pkg.build(\"Bosque\"), and restart Julia.")
    end
end
