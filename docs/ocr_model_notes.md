# OCR / Ingestion Module — Model Notes (Week 6)

## Components
- **Preprocessing:** OpenCV — denoise, deskew, adaptive-threshold binarize.
- **Printed text:** Tesseract (pytesseract). Tested on a real invoice
  image: 77% average word confidence, fully readable extracted text
  with minor expected noise (e.g. "QUANTHY" for "QUANTITY").
- **Handwritten text:** Gemini API (gemini-3.6-flash), multimodal.

## Tooling substitution: PaddleOCR -> Tesseract + TrOCR (+ later, Gemini)
Plan originally specified PaddleOCR. PaddleOCR has historically poor,
inconsistent support on Apple Silicon — substituted Tesseract (printed)
+ initially TrOCR (handwritten) as a same-goal, better-fit-for-environment
swap, same reasoning as the Week 4 OpenAI -> Groq substitution.

## TrOCR attempt and why it was replaced
TrOCR (microsoft/trocr-base-handwritten) was tried first for handwriting.
Problems encountered, in order:

1. **Tokenizer loading failure** on this transformers version — worked
   around via explicit ViTImageProcessor/RobertaTokenizer loading.
2. **Severe truncation** (default max_length=21) — fixed via
   max_new_tokens=128.
3. **Whole-document input mismatch**: TrOCR is trained on single
   cropped text lines, not full documents. Feeding it a whole
   prescription image produced 2-3 words of real output only.
   Built modules/ocr/line_segmentation.py (OpenCV horizontal
   projection profiling) to split documents into individual lines
   before OCR, matching TrOCR's actual training distribution.
4. **Repetition loops** on real, messy handwriting (prescription
   sample) — the model would get stuck generating "2 2 2 2 2..." or
   "0002000200020002...". Fixed via repetition_penalty=2.5 and
   no_repeat_ngram_size=3.
5. **Fluent hallucination**, even on a controlled, clearly-written test
   sentence written specifically to isolate the problem: TrOCR produced
   grammatically fluent but entirely unrelated text ("transmembrancers
   and their success were also considered in the summerwives...").
   Confidence dropped to 0.14 — an honest signal the model itself
   didn't trust its own output, but the output was still unusable.

This progression (each fix revealing the next failure mode) is a
realistic pattern in ML engineering, not a sign of a broken approach —
but ultimately indicated TrOCR-base is not fit for this use case
without significant additional work (proper line-segmentation tuning,
and likely fine-tuning on domain-specific — e.g. prescription-style —
handwriting data), which is out of scope for this project.

## Pivot to Gemini
Switched to Gemini's multimodal API (gemini-3.6-flash) for handwriting:
sends the whole document image directly, no line segmentation needed,
using a broader vision-language model rather than a narrow single-line
specialist. Result: accurate extraction on the same test cases that
caused TrOCR to hallucinate. modules/ocr/line_segmentation.py was
removed as dead code once no longer needed.

## Confidence caveat
Gemini does not expose token-level log-probabilities via this API
the way a raw model would. Confidence is self-reported by the model
via structured JSON prompting ("rate your confidence 0.0-1.0") — a
practical technique, but not a mathematically calibrated probability
the way Tesseract's per-word confidence or Vision/Language's softmax
outputs are. Documented explicitly for the same reason as Whisper's
(Week 5) and TrOCR's approximated confidence scores.

## Next steps (future work, not blocking Week 6 completion)
- Currently Tesseract (printed) and Gemini (handwritten) are separate
  functions; need a routing layer (using Vision's doc_type output) to
  automatically choose the right one per document.
- Consider using Gemini for printed text too, and comparing its
  accuracy/cost against Tesseract's free, local, already-strong result.

## Bug found and fixed: silent segfault from faiss/torch import order
modules/ocr/ingest.py combines faiss (Language module) with Vision's
torch-based predict() in one process for the first time. Initial
version crashed with "zsh: segmentation fault" and NO Python traceback
at all — immediately after logging "Ingesting <filename>", before any
of our own code's debug output even printed.

Debugging process: isolated each step (Vision alone: worked fine; PIL
opening the specific .webp file alone: worked fine), ruling out the
image format and Vision's logic individually. The segfault only
occurred when faiss was imported in the same process as torch-based
Vision code — a known class of issue: both libraries bundle their own
native OpenMP runtime, and depending on import order, they can
conflict at the C++ level in a way Python's exception handling can't
catch (hence: total silence, no traceback, just a crash).

Fix: (1) set KMP_DUPLICATE_LIB_OK=TRUE as a mitigation, (2) import
faiss AFTER torch-dependent modules (Vision) rather than at the top
of the file. Both together resolved it. Verified end-to-end: a real
prescription image was classified (Vision), OCR'd (Gemini, 0.98
confidence), embedded, and added to the FAISS index successfully, then
correctly retrieved and answered via the full voice_chat.py pipeline.

This is a good example of a bug that only appears at genuine
integration time — each module was correctly tested in isolation
(Weeks 3, 4, 6), but combining them for the first time surfaced a
real, environment-specific conflict that no single module's tests
could have caught.