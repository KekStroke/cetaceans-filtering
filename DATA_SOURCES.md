# Data sources

Source audio is downloaded at run time and is not distributed by this
repository. The repository's MIT license does not replace the terms of the
datasets below.

| Source | Access used by this repository | Rights and citation |
| --- | --- | --- |
| Watkins Marine Mammal Sound Database | Hugging Face mirror [`confit/wmms-parquet`](https://huggingface.co/datasets/confit/wmms-parquet), via `utils/datasets_downloads/download_watkins.py` | The dataset card permits personal or academic, non-commercial use and requests credit to the Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution, and the New Bedford Whaling Museum. Check the current dataset card before use. |
| NOAA SanctSound / passive bioacoustics | Public `noaa-passive-bioacoustic` bucket, via `utils/datasets_downloads/download_noaa_onms.py` | Cite the deployment metadata; the project citation used here is [doi:10.25921/saca-sp25](https://doi.org/10.25921/saca-sp25). |
| Orcasound | AWS Open Data `acoustic-sandbox` sources, via `utils/datasets_downloads/download_orcasound.py` | [AWS registry entry](https://registry.opendata.aws/orcasound/): CC BY-NC-SA 4.0; its requested citation includes the access date. |
| Pacific Ocean Sound Recordings | AWS buckets for the original 256 kHz and decimated 16 kHz/2 kHz archives, via `utils/datasets_downloads/download_pacific_sound.py` | [AWS registry entry](https://registry.opendata.aws/pacific-sound/): CC BY 4.0; its requested citation includes the access date. |
| Voices in the Sea | [UC San Diego audio endpoint](https://voicesinthesea.ucsd.edu/species/spectrogramPlayerComponents/audio/), via `utils/datasets_downloads/download_voices_in_the_sea.py` | This repository does not record a reuse license for this source. Check the provider's current terms before use or redistribution. |
| Ocean Networks Canada | Oceans 3.0 API, via `utils/datasets_downloads/download_onc_hydrophones.py`; an `ONC_TOKEN` is required | Follow the [ONC data policy](https://www.oceannetworks.ca/data/data-policy/) and the rights attached to each requested deployment. ONC-owned data are CC BY 4.0; partner-owned data can have different terms. Use the dataset-specific DOI or Query PID citation. |

Exact corpus membership depends on the configuration and the remote listing at
download time. For a reproducible snapshot, preserve the Hydra configuration
and overrides, retrieval date, generated manifests, and manifest SHA-256
checksums. Never commit API tokens or restricted source audio. See
[`docs/animal2vec_dataset_download/README.md`](docs/animal2vec_dataset_download/README.md)
for the corpus assembly workflow.
