# Input structures

Place structures for conversational MACE energy calculations in this directory.

Supported names and formats:

- `*.cif`
- `*.vasp`
- `*.poscar`
- `POSCAR`
- `POSCAR_*` or `POSCAR-*`

Example request:

```text
我要计算这个 graphene.cif 的能量
```

The calculation is a MACE single-point potential energy for the structure as
provided. It does not relax the atoms or cell. User structure files in this
directory are ignored by Git; only this README is tracked.
