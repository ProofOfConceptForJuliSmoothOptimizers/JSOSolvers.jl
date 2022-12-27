#!/bin/bash

export tuningFile="$1"
export org="$2"
export repo="$3"
export pullrequest="$4"
julia --project=tuning -E 'using Pkg; Pkg.instantiate(); Pkg.resolve()'
julia --project=tuning tuning/${tuningFile} &> "$org"_"$repo"_"$pullrequest".txt
julia --project=tuning tuning/send_gist_url.jl "$org" "$repo" "$pullrequest"
