# UT TRAIN description review export

CPU only; Python 3.8 compatible; uses existing torch and Pillow.

Run export_train_review.py with --data, --knowledge and --out.
Version 3 adds --image-field _image for the original UT image tree. UT metadata
has two paths: image (CPF composition/sample ID) and _image (original file path).
The image ID and sorting/sampling remain unchanged. Only source_relative_path
is used to open the original file; both fields and the selected path mode are
saved for provenance. Missing _image, mismatched basenames, or missing original
files are rejected without guessing filenames or replacing selected images.
--image-root explicitly chooses the real image tree and
--check-images-only to check every selected TRAIN path without creating output.
The default root is DATA/images. Existing external directory links are not
automatically trusted: set --image-root to the verified tree containing the
metadata-relative Boots/, Shoes/, and Sandals/ paths. The real root is recorded
in manifest.json. Path escapes and missing files report different errors.
Sampling identities and the fixed seed are unchanged from versions 1 and 2.
The metadata file contains all splits, but records are filtered to set=train
before accessing image paths. Only train_pairs.txt is read. No validation/test
images, model checkpoints, GloVe scan, training, or semantic gate changes.

Fixed default seed: 20261006. Maximum 20 panels per class, balanced across the
other primitive. The balanced sample is for qualitative coverage, not for
estimating dataset prevalence or performance. Repeat panels across class boards
are not independent observations. Images retain their whole silhouette.

Outputs:
- boards/: one JPEG contact sheet per class
- selected_train_images.json: image identities, TRAIN labels, source hashes
- review_template.md: all 84 original descriptions with pending review fields
- manifest.json: source fingerprints and sampling scope
- A ZIP next to the output folder, for transfer to the local review computer

All commands and terminal output are English. Existing output paths are rejected.
No approval is inferred from generating a review template.

check_export.py is a synthetic CPU regression check. It creates its own fixture
in a tmp folder, does not use the real dataset, and is not required for export.
The calling environment needs this directory on sys.path to run that check.

See the accompanying Chinese Step 7 document for the exact server and Windows
transfer commands, experiment budgets, and continue/stop decisions.
