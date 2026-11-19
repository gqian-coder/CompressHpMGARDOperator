# CompressHpMGARDOperator

Error-bounded compression of high-order DG/FR solution arrays (hpMusic / Fides `cell_grid` layout,
`[cell][solution point][component]`) and an ADIOS2 plugin operator around it.

The library (`hpmgard`) splits each component into a low-order base (Gauss L2 projection, stored at the
cell corners so the prolongation interpolates) and a residual taken against the decoded base, quantises
both under the requested bound and codes them with chainable lossless stages (`raw`, `frbp`, `huffman`,
`zstd:<level>`, optional `mgardx`). An auto-tuner picks the base order, base quantum, base split and codecs
per block from a sample of cells. Every setting is written into a self-describing, CRC-protected header,
so decompression needs no parameters.

Status: v1, compression only, CPU (OpenMP) reference implementation. HIP and CUDA backends, and the
progressive-retrieval extension, come later. Tools, tests, drivers and documents live in the companion
repository `HpMGARD-Tools`.

## Build (CPU)

```bash
mkdir build_cpu && cd build_cpu && source ../runconf_cpu && make -j16 && cmake --install .
```

Options: `HPMGARD_ENABLE_MGARDX` (default ON if MGARD-X is found), `HPMGARD_BUILD_ADIOS2_PLUGIN`.

## Layout

```
src/hpmgard/          core library: Types, Quadrature, Operators (tensor sweeps + ported simplex tables),
                      BaseSplit (cell average, vertex dedup + Hilbert order), codec/ (Raw, FRBP, Huffman,
                      Zstd, chain container), MgardX wrapper, Header, Compressor, Tuner
src/hpmgard/ported/   files copied verbatim from mp-MGARD (see PROVENANCE.md)
src/operator/         ADIOS2 plugin operator (CompressHpMGARDOperator, MeshAdapter)
```
