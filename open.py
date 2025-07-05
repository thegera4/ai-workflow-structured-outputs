# This is a multimodel and multistep AI workflow to generate engaging posts for X (Twitter) based on a website content.
# This script fetches a website's HTML, extracts the core content, summarizes it, and generates a post for X.
# This script uses different Gemma AI open source models to perform the tasks.

import os
import json
import sys
import sqlite3
import requests
from pyexpat.errors import messages
from pypdf import PdfReader
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class Vendor(BaseModel):
    name: str = Field(..., description="The name of the vendor or company issuing the invoice.")
    address: str = Field(..., description="The address of the vendor.")
    phone: str = Field(..., description="The phone of the vendor.")
    email: str = Field(..., description="The email of the vendor.")


class Customer(BaseModel):
    name: str = Field(..., description="The name of the customer or client.")
    address: str = Field(..., description="The address of the customer.")
    phone: str = Field(..., description="The phone of the customer.")
    email: str = Field(..., description="The email of the customer.")


class Invoice(BaseModel):
    vendor: Vendor = Field(..., description="Details of the vendor issuing the invoice.")
    customer: Customer = Field(..., description="Details of the customer receiving the invoice.")
    invoiceNumber: str = Field(..., description="Unique identifier for the invoice.")
    date: str = Field(..., description="Date when the invoice was issued.")
    totalAmount: float = Field(..., description="Total amount due on the invoice.")
    tax: float = Field(..., description="Total tax amount applied to the invoice.")
    paymentTerms: int = Field(..., description="Payment terms for the invoice, such as 'Net 30' or 'Due on receipt'.")


def setup_database():
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY,
            vendor_name TEXT,
            vendor_address TEXT,
            vendor_phone TEXT,
            vendor_email TEXT,
            customer_name TEXT,
            customer_address TEXT,
            customer_phone TEXT,
            customer_email TEXT,
            invoice_number TEXT,
            date TEXT,
            total_amount REAL,
            tax REAL,
            payment_terms INTEGER
        )
    ''')
    conn.commit()
    return conn


def insert_invoice_data(conn, invoice_data):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO invoices (
            vendor_name, vendor_address, vendor_phone, vendor_email,
            customer_name, customer_address, customer_phone, customer_email,
            invoice_number, "date", total_amount, tax, payment_terms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        invoice_data.get("vendor", {}).get("name"),
        invoice_data.get("vendor", {}).get("address"),
        invoice_data.get("vendor", {}).get("phone"),
        invoice_data.get("vendor", {}).get("email"),
        invoice_data.get("customer", {}).get("name"),
        invoice_data.get("customer", {}).get("address"),
        invoice_data.get("customer", {}).get("phone"),
        invoice_data.get("customer", {}).get("email"),
        invoice_data.get("invoiceNumber"),
        invoice_data.get("date"),
        invoice_data.get("totalAmount"),
        invoice_data.get("tax"),
        invoice_data.get("paymentTerms")
    ))
    conn.commit()


def get_pdf_content(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


expected_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "invoice_details",
        "strict": "true",
        "schema": {
            "type": "object",
            "properties": {
                "vendor": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "phone": {"type": "string"},
                        "email": {"type": "string"}
                    },
                    "required": ["name", "address", "phone", "email"]
                },
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "phone": {"type": "string"},
                        "email": {"type": "string"}
                    },
                    "required": ["name", "address", "phone", "email"]
                },
                "invoiceNumber": {"type": "string"},
                "date": {"type": "string"},
                "totalAmount": {"type": "number"},
                "tax": {"type": "number"},
                "paymentTerms": {"type": "integer"}
            },
            "required": ["vendor", "customer", "invoiceNumber", "date",
                         "totalAmount", "tax", "paymentTerms"]
        }
    }
}


def get_ai_response(model: str, role: str, prompt: str, schema: dict, temp: float, ctx: int = 4000) -> str:
    msgs = [
        {"role": "system", "content": role},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
            model=model,
            messages=msgs,
            response_format=schema
    )

    results = json.loads(response.choices[0].message.content)
    return json.dumps(results, indent=2) if isinstance(results, dict) else results


def extract_invoice_details(pdf_content: str) -> Invoice:
    prompt = f"""
        Extract all relevant data from the below invoice content (which was extracted from a PDF document).
        Make sure to capture data like vendor name, date, amount, tax etc.
        <invoice-content>
        {pdf_content}
        </invoice-content>
        
        Return your response as a JSON object without any extra text or explanation.
    """

    response = get_ai_response(
        model="google/gemma-3-12b",
        role="You are an expert data extractor who excels at analyzing invoices.",
        prompt=prompt,
        schema=expected_schema,
        temp=0.5,
        ctx=20000
    )

    if not response:
        raise ValueError("Failed to extract invoice details from the PDF content.")

    try:
        invoice_data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")

    # Unpack the 'invoice' object and merge with vendor/customer
    if 'invoice' in invoice_data:
        merged = {**invoice_data['invoice'], 'vendor': invoice_data['vendor'], 'customer': invoice_data['customer']}
        return Invoice(**merged)
    else:
        return Invoice(**invoice_data)


def main():
    if len(sys.argv) < 2:
        print("Usage: python open.py /path/to/file_or_folder")
        return

    path = sys.argv[1]
    pdf_files = []

    if not os.path.exists(path):
        print(f"Error: The path '{path}' does not exist.")
        return

    if os.path.isfile(path):
        if path.lower().endswith(".pdf"):
            pdf_files.append(path)
        else:
            print(f"Error: The file '{path}' is not a PDF file.")
            return
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(path, filename))

    if not pdf_files:
        print("No PDF files found.")
        return

    conn = setup_database()

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        try:
            pdf_content = get_pdf_content(pdf_file)
            invoice = extract_invoice_details(pdf_content)
            insert_invoice_data(conn, invoice.model_dump())
            print("Extracted Invoice Details:")
            print(invoice.model_dump())
            print("---------")
        except Exception as e:
            print(f"An error occurred while processing {pdf_file}: {e}")

    conn.close()


if __name__ == "__main__":
    main()