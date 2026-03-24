# London Commuting Model 
This repository contains an agent-based model of commuting patterns in London.  

# Data Sources
MSOA boundary: https://geoportal.statistics.gov.uk/
Cogestion data: https://www.tomtom.com/downloads/traffic-index/#downloads
Commute mode: https://www.nomisweb.co.uk/api/v01/dataset/NM_568_1.bulk.csv?time=2011&geography=TYPE297&measures=20100
OD travel2work: https://doi.org/10.5281/zenodo.13327082
MSOA occupation: https://www.nomisweb.co.uk/sources/census_2021_bulk

## Run the Model
```bash
solara run app.py