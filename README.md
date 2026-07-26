# sq-diagrams

Draw antisymmetrized-Goldstone (Brandow) diagrams for SeQuant coupled-cluster terms.

## Build

    cmake -S . -B build && cmake --build build --target sq-diagram-extract

That fetches and builds SeQuant once. To reuse an existing SeQuant build tree
instead, point `CMAKE_PREFIX_PATH` at it — plus its dependency build/install
trees, since SeQuant's build-tree export references them by name:

    SQ=~/Code/SeQuant/build/relwithdebinfo
    cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_PREFIX_PATH="$SQ;~/Code/dep-installs/range-v3;$SQ/_deps/libperm-build;$SQ/_deps/polymorphic_variant-build;/opt/homebrew/opt/eigen@3;/opt/homebrew/opt/boost"

## Use

    build/sq-diagram-extract "g{i1,i2;a1,a2} t{a2,a3;i1,i2}" | python sq-diagrams/draw.py - out.svg

Extractor prints diagram topology as JSON; `draw.py` renders it. See
`brandow-diagram-generator-design.md` (in ~/Notes/diagrams) for the design.

## Test

    pip install -r requirements.txt && python -m pytest tests/ -v

## Scope (v1)

One connected CCSD-level term (tensors `f`/`g`/`t` of rank ≤ 2) to one image.
Batch rendering over a `Sum`, crossing minimization, CCSDT+, and disconnected
terms are not implemented.
