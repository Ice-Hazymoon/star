# Data Sources

This project keeps the recognition pipeline and the display vocabulary separate.

## Canonical astronomy sources

These sources define object identity, coordinates, or accepted naming:

- [IAU constellations](https://www.iau.org/IAU/Iau/Science/What-we-do/The-Constellations.aspx)
  - authoritative source for the 88 modern constellations and their boundaries
  - does **not** define the familiar stick-figure line art
- [IAU Working Group on Star Names](https://www.iau.org/WG280/WG280/Home.aspx)
  - authoritative source for approved stellar proper names

## Runtime catalog sources in this repo

- `data/catalog/minimal_hipparcos.csv`
  - reduced Hipparcos subset used for fast coordinate lookup during solving and overlay assembly
- `data/reference/modern_st.json`
  - Stellarium modern sky-culture constellation data
  - used for constellation membership and HIP-based line topology
- `data/reference/common_star_names.fab`
  - Stellarium common star-name table
  - used for named-star labeling
- `data/reference/NGC.csv`
  - OpenNGC-style deep-sky catalog distributed with Astrometry.net
  - used as the broad DSO base catalog

## Supplemental visual / localization sources

- `data/reference/stardroid-constellations.ascii`
  - copied from Stardroid
  - used for richer constellation line segments and label anchor positions
- `data/reference/stardroid-deep_sky_objects.csv`
  - copied from Stardroid
  - used for curated Messier and notable DSO aliases
- `data/reference/stardroid-locales/*/celestial_objects.xml`
  - copied from Stardroid Android resources
  - currently bundles 30 locale variants including `en`, `en-GB`, `ja`, `fr`, `de`, `zh-Hans`, `zh-Hant`, `ru`, and others
  - used as the primary maintained localization source for constellation names and part of the DSO label set
  - request-time locale selection always falls back to the bundled English source table instead of hand-written translations
- `data/reference/supplemental-deep-sky-objects.json`
  - project-owned supplement for objects not covered cleanly by the main catalogs
  - currently limited to the Hyades cluster so the runtime stays small and traceable

## Important caveat

Constellation line art is not an IAU-standard dataset. In practice, planetarium apps and sky-mapping tools use curated line sets. This project uses:

1. Stellarium for the main constellation topology
2. Stardroid for denser visual line segments and better label anchors

That combination is deliberate: it keeps object identity tied to established sources while improving the rendered overlay.
