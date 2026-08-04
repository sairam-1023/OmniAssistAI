"""
Sample document text, standing in for real OCR output (which arrives
in Week 6). Each entry represents one "document" with realistic
content, used to build and test the RAG pipeline end-to-end before
real OCR text exists.
"""

SAMPLE_DOCUMENTS = [
    {
        "document_id": "doc_001",
        "filename": "acme_invoice_march.png",
        "doc_type": "invoice",
        "text": (
            "INVOICE #INV-20340\n"
            "Acme Corp, 500 Industrial Way, Springfield, IL\n"
            "Bill To: Northwind Traders, 12 Market St, Chicago, IL\n"
            "Invoice Date: March 3, 2026\n"
            "Due Date: April 2, 2026\n"
            "Items:\n"
            "  - Office Chairs (x10) ... $1,200.00\n"
            "  - Standing Desks (x4) ... $2,400.00\n"
            "  - Shipping ... $150.00\n"
            "Subtotal: $3,750.00\n"
            "Tax (8%): $300.00\n"
            "Total Due: $4,050.00\n"
            "Payment Terms: Net 30\n"
            "Status: UNPAID"
        ),
    },
    {
        "document_id": "doc_002",
        "filename": "coffee_receipt.png",
        "doc_type": "receipt",
        "text": (
            "Blue Bottle Coffee\n"
            "221 Baker St, San Francisco, CA\n"
            "Date: 2026-03-15  Time: 08:42 AM\n"
            "1x Cappuccino ... $4.50\n"
            "1x Croissant ... $3.75\n"
            "Subtotal: $8.25\n"
            "Tax: $0.68\n"
            "Total: $8.93\n"
            "Payment: Visa ending 4021\n"
            "Thank you for visiting!"
        ),
    },
    {
        "document_id": "doc_003",
        "filename": "rx_amoxicillin.png",
        "doc_type": "prescription",
        "text": (
            "Springfield Medical Group\n"
            "Patient: Jane Doe, DOB: 1990-04-12\n"
            "Prescribing Physician: Dr. Ahmed Khan\n"
            "Date Issued: 2026-02-20\n"
            "Medication: Amoxicillin 500mg\n"
            "Dosage: Take 1 capsule by mouth 3 times daily for 10 days\n"
            "Quantity: 30 capsules\n"
            "Refills: 0\n"
            "Pharmacy Notes: Take with food. Complete full course even if "
            "symptoms improve."
        ),
    },
    {
        "document_id": "doc_004",
        "filename": "patient_intake_form.png",
        "doc_type": "form",
        "text": (
            "New Patient Intake Form\n"
            "Full Name: Michael Chen\n"
            "Date of Birth: 1985-11-02\n"
            "Insurance Provider: BlueCross BlueShield\n"
            "Policy Number: BC-88213764\n"
            "Emergency Contact: Lisa Chen, (555) 019-2233\n"
            "Reason for Visit: Annual physical exam\n"
            "Allergies: Penicillin\n"
            "Signature: M. Chen\n"
            "Date Submitted: 2026-01-10"
        ),
    },
    {
        "document_id": "doc_005",
        "filename": "electric_bill_invoice.png",
        "doc_type": "invoice",
        "text": (
            "INVOICE #PWR-99881\n"
            "Springfield Electric Utility\n"
            "Account Holder: Sarah Johnson\n"
            "Service Address: 44 Oak Lane, Springfield, IL\n"
            "Billing Period: Feb 1 - Feb 28, 2026\n"
            "Usage: 620 kWh\n"
            "Rate: $0.14/kWh\n"
            "Total Due: $86.80\n"
            "Due Date: March 20, 2026\n"
            "Status: PAID on 2026-03-05"
        ),
    },
    {
        "document_id": "doc_006",
        "filename": "drivers_license.png",
        "doc_type": "id_card",
        "text": (
            "STATE OF ILLINOIS - DRIVER LICENSE\n"
            "Name: Robert Alan Taylor\n"
            "DOB: 1978-06-30\n"
            "License Number: T-4821-9930-2201\n"
            "Address: 78 Willow Ave, Chicago, IL\n"
            "Issued: 2023-06-30\n"
            "Expires: 2028-06-30\n"
            "Class: D"
        ),
    },
]