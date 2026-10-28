# Data attribution and rights separation

No source recordings or cleaned motion datasets are distributed in this repo.
WhoIsAlyx data are supplied by their creators under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), independently
of any eventual software license chosen for this code.

Primary citation:

Rack, C., Fernando, T., Yalcin, M., Hotho, A., and Latoschik, M. E. (2023).
**Who is Alyx? A new behavioral biometric dataset for user identification in XR.**
Frontiers in Virtual Reality 4. [DOI 10.3389/frvir.2023.1272234](https://doi.org/10.3389/frvir.2023.1272234).

Source repository pinned to
[5da510121f99fc8a6cc51e570770ca93e5745646](https://github.com/cschell/who-is-alyx/tree/5da510121f99fc8a6cc51e570770ca93e5745646).
The [README](https://github.com/cschell/who-is-alyx/blob/5da510121f99fc8a6cc51e570770ca93e5745646/Readme.md)
and [license file](https://github.com/cschell/who-is-alyx/blob/5da510121f99fc8a6cc51e570770ca93e5745646/License.md)
contain creator notices that should both be preserved with privately downloaded
data; their creator-name lists differ. The acquisition stage caches the original
files rather than silently rewriting their notices.

Pipeline transformations include HMD-column extraction, removal of invalid rows,
normalization, sign alignment, session filtering, duration cap, gap splitting and
offline SLERP. If adapted data are later shared lawfully, retain attribution,
license/modification notices and applicable ShareAlike terms. This project does
not make a legal determination about trained-model licensing or recipient
institutional obligations. See the source audit for unresolved gates.
