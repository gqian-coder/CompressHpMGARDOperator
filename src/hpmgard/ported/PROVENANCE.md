# Ported files

Copied verbatim from mp-MGARD (https://github.com/gqian-coder/mp-MDR), commit 3daa59457cf0d408eab2216cd66d94f0352bca67, on 2026-11-19.
Namespaces (pmgpu, mp_mgard::sfc) are kept so the files stay byte-identical to the source; they will be
renamed in one commit once the port is complete.

| file | source path | role |
|---|---|---|
| rp_operators.h / .cpp | src/core/ | DenseMatrix, Gauss-Legendre tensor R/P tables (used for the dense simplex path's matrix type) |
| gauss_quadrature.h, gll_quadrature.h, tri_quadrature.h | src/core/ | node/weight tables used by the tables below |
| tet_rp_operators.h / .cpp | src/core/ | dense TET restriction/prolongation (hpMusic node set) |
| prism_rp_operators.h / .cpp | src/core/ | dense PRISM restriction/prolongation |
| pyramid_rp_operators.h / .cpp | src/core/ | dense PYRAMID restriction/prolongation |
| sfc.h | include/mp_mgard/core/ | Hilbert key and coordinate quantisation (host) |
a2636e4ff625ac7b23fd1d5e7a496a616b426f674208e284e16a71058710d754  gauss_quadrature.h
3d471941f8b74c110e3c2ded9880fa7f58ead374d5cb67d0a2e0331c59988832  gll_quadrature.h
4e7c6bf8efe94f3bd795a52ac017c5b77c292e4f914f6d5043fd4e7dc43672a3  prism_rp_operators.h
c2a754b187adba29a445ca69d4c79fc7c1a476591e020590ee874832d2c0cf71  pyramid_rp_operators.h
1cbb69ffce80950970ee20a808ca3840b9e3d92e4c0abe240196198b453e454f  rp_operators.h
5db929397c281dc9e972ff7e42940ee8baf4dc3367b0ca8df17b72090129edb2  sfc.h
1e76abf414137f7c141626b8fbb355d82111d2b2286c8a2e5f9072edd5a59139  tet_rp_operators.h
6a426a76ed53f6cda204bc487a66dfc647b1d134750263358b64807d0a2b043e  tri_quadrature.h
57f30d894191c8406b935409588d28381f3842c27f243c7b991fc3dff7331093  prism_rp_operators.cpp
0929eda2dd525f5196eb62ddee4aa44e1c5f4e2f266fec32777b082b99012fcc  pyramid_rp_operators.cpp
7d8e974e1ef790375ba78de56c5226cff17a91fdeb834f58e202486a424e80ff  rp_operators.cpp
2bc3ad0f1332811d00627b9a2939f731b5918979efee44720e0abd0dda78c5d7  tet_rp_operators.cpp
