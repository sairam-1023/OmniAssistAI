# Platform Layer — Notes (Week 7)

## Known limitation: no persistent link from document_id to original file
The vector index (chunks.pkl / metadata.json) stores extracted text and
a filename label, but not a reliable path back to the original image
file. Uploaded files are saved to storage/uploads/ under randomized
UUID names; the original filename is preserved only as a metadata
label, disconnected from the actual saved file location. The 6
original Week 4 mock documents never existed as real image files at
all (their text was hand-written directly in Python).

Impact: no current way to, e.g., let a user click "view original
document" from a search result and see the actual uploaded image.

Not fixed in Week 7 — would require a proper document registry (at
minimum, a JSON/DB table mapping document_id -> stored file path),
which is a reasonable v1.1 addition rather than a blocker for getting
the API live.