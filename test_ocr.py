from modules.vision.predict import predict as classify_document
from modules.ocr.extract import extract_text

image_path = "data/real_images/invoice/aF0Dc.jpg"  # use a real one
vision_result = classify_document(image_path)
print(f"Vision classified as: {vision_result['doc_type']}")

ocr_result = extract_text(image_path, vision_result["doc_type"])
print(f"OCR engine used: {ocr_result['engine']}")
print(f"Extracted text: {ocr_result['text']}")
print(f"Confidence: {ocr_result['confidence']}")