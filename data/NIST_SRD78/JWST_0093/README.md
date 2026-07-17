# JWST_0093 NIST SRD 78 data

Official query-specific line tables downloaded from the NIST Atomic Spectra Database.

- Database: NIST Standard Reference Database 78
- Database version recorded by script: 5.12
- Wavelength convention: vacuum below 200 nm
- Six selected ionic families: N IV], C IV, He II, O III], N III], C III]
- Downloaded UTC: 2026-07-17T00:25:56+00:00

## Files

- `*_FULL_TABLE.csv`: complete parsed NIST response table
- `*_CANONICAL.csv`: normalized wavelength and strength columns
- `JWST_0093_NIST_SRD78_SIX_REFERENCE_FAMILIES_MASTER.csv`: merged canonical table
- `JWST_0093_NIST_SRD78_QUERY_MANIFEST.csv`: query and provenance audit
- `JWST_0093_NIST_SRD78_SIX_REFERENCE_FAMILIES.xlsx`: workbook with all tables
- `JWST_0093_NIST_SRD78_RAW_RESPONSES.zip`: exact server responses

## Row counts

```text
family  parsed_table_rows  canonical_wavelength_rows parse_mode
 N IV]                  2                          2        TAB
  C IV                  2                          2        TAB
 He II                  7                          7        TAB
O III]                  9                          9        TAB
N III]                 16                         16        TAB
C III]                  9                          9        TAB
```
