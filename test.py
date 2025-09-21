import cv2
import pytesseract
from PIL import Image
import numpy as np
import re
import streamlit as st
from typing import Dict, List

class ReceiptTextExtractor:
    def __init__(self):
        # Configure tesseract path if needed (Windows users)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess the image to improve OCR accuracy
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Optional: Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text_basic(self, image_bytes: bytes) -> str:
        """
        Basic text extraction from receipt image
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_bytes)
            
            # Configure OCR settings for better results
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,$/():@*-+ '
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            return text.strip()
        
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
    def extract_text_with_coordinates(self, image_bytes: bytes) -> List[Dict]:
        """
        Extract text with bounding box coordinates
        """
        try:
            processed_img = self.preprocess_image(image_bytes)
            
            # Get detailed OCR data including coordinates
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            
            extracted_data = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Filter low confidence text
                    text = data['text'][i].strip()
                    if text:  # Only include non-empty text
                        extracted_data.append({
                            'text': text,
                            'confidence': data['conf'][i],
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        })
            
            return extracted_data
        
        except Exception as e:
            st.error(f"Error extracting text with coordinates: {str(e)}")
            return []
    
    def identify_walmart_receipt(self, text: str) -> Dict:
        """
        Specifically identify if this is a Walmart receipt and extract Walmart-specific data
        """
        text_lower = text.lower()
        
        walmart_indicators = [
            'walmart',
            'wal*mart',
            'wal-mart',
            'save money. live better',
            'supercenter',
            'neighborhood market'
        ]
        
        # Check for Walmart indicators
        is_walmart = any(indicator in text_lower for indicator in walmart_indicators)
        
        walmart_data = {
            'is_walmart': is_walmart,
            'confidence_score': 0,
            'walmart_indicators_found': [],
            'store_format': '',
            'manager_info': '',
            'associate_info': ''
        }
        
        if is_walmart:
            # Calculate confidence score based on indicators found
            indicators_found = [indicator for indicator in walmart_indicators if indicator in text_lower]
            walmart_data['walmart_indicators_found'] = indicators_found
            walmart_data['confidence_score'] = len(indicators_found) * 20  # Max 100%
            
            # Identify store format
            if 'supercenter' in text_lower:
                walmart_data['store_format'] = 'Supercenter'
            elif 'neighborhood market' in text_lower:
                walmart_data['store_format'] = 'Neighborhood Market'
            elif 'walmart' in text_lower:
                walmart_data['store_format'] = 'Standard Walmart'
            
            # Look for manager info
            lines = text.split('\n')
            for line in lines:
                if 'mgr' in line.lower() or 'manager' in line.lower():
                    walmart_data['manager_info'] = line.strip()
                    break
            
            # Look for associate info
            for line in lines:
                if 'op#' in line.lower() or 'operator' in line.lower():
                    walmart_data['associate_info'] = line.strip()
                    break
        
        return walmart_data
    
    def parse_walmart_specific_data(self, text: str) -> Dict:
        """
        Extract Walmart-specific information from receipt
        """
        lines = text.split('\n')
        walmart_specific = {
            'store_number': '',
            'terminal_number': '',
            'transaction_number': '',
            'operator_id': '',
            'manager_name': '',
            'store_address': '',
            'phone_number': '',
            'tax_id': ''
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Store number (usually after ST#)
            if 'st#' in line_lower or 'store#' in line_lower:
                store_match = re.search(r'st#?\s*(\d+)', line_lower)
                if store_match:
                    walmart_specific['store_number'] = store_match.group(1)
            
            # Terminal number (TE#)
            if 'te#' in line_lower:
                terminal_match = re.search(r'te#?\s*(\d+)', line_lower)
                if terminal_match:
                    walmart_specific['terminal_number'] = terminal_match.group(1)
            
            # Transaction number (TR#)
            if 'tr#' in line_lower:
                trans_match = re.search(r'tr#?\s*(\d+)', line_lower)
                if trans_match:
                    walmart_specific['transaction_number'] = trans_match.group(1)
            
            # Operator ID (OP#)
            if 'op#' in line_lower:
                op_match = re.search(r'op#?\s*(\d+)', line_lower)
                if op_match:
                    walmart_specific['operator_id'] = op_match.group(1)
            
            # Manager info
            if 'mgr' in line_lower:
                walmart_specific['manager_name'] = line.strip()
            
            # Phone number
            phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', line)
            if phone_match and not walmart_specific['phone_number']:
                walmart_specific['phone_number'] = phone_match.group()
            
            # Address (look for city, state pattern)
            if re.search(r'[A-Z]{2}\s+\d{5}', line) or any(city in line_lower for city in ['opelika', 'al']):
                walmart_specific['store_address'] = line.strip()
        
        return walmart_specific
    
    def parse_receipt_structure(self, text: str) -> Dict:
        """
        Parse extracted text to identify receipt components
        """
        lines = text.split('\n')
        receipt_data = {
            'store_name': '',
            'address': '',
            'phone': '',
            'date': '',
            'time': '',
            'items': [],
            'subtotal': '',
            'tax': '',
            'total': '',
            'payment_method': ''
        }
        
        # Patterns for different receipt components
        price_pattern = r'\$?\d+\.\d{2}'
        date_pattern = r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}'
        time_pattern = r'\d{1,2}:\d{2}(?::\d{2})?'
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        # Improved patterns for weighted items
        weight_pattern = r'\d+\.\d+\s*lb\s*@\s*\d+\.\d+\s*lb\s*[/\\]\s*\d+\.\d+'
        
        # Pattern for bulk pricing (like 3 AT 1 FOR 0.83)
        bulk_pricing_pattern = r'\d+\s*AT\s*\d+\s*FOR\s*\d+\.\d{2}'
        
        # Pattern to identify product lines (has product code and F at end)
        product_code_pattern = r'[A-Z\s]+\d{12}\s*F'
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Extract store name (usually first non-empty line)
            if not receipt_data['store_name'] and i < 3:
                if not re.search(r'\d', line) and len(line) > 3:
                    receipt_data['store_name'] = line
            
            # Extract date
            date_match = re.search(date_pattern, line)
            if date_match and not receipt_data['date']:
                receipt_data['date'] = date_match.group()
            
            # Extract time
            time_match = re.search(time_pattern, line)
            if time_match and not receipt_data['time']:
                receipt_data['time'] = time_match.group()
            
            # Extract phone number
            phone_match = re.search(phone_pattern, line)
            if phone_match and not receipt_data['phone']:
                receipt_data['phone'] = phone_match.group()
            
            # Extract totals
            if 'subtotal' in line.lower():
                price_match = re.search(price_pattern, line)
                if price_match:
                    receipt_data['subtotal'] = price_match.group()
            
            elif 'tax' in line.lower():
                price_match = re.search(price_pattern, line)
                if price_match:
                    receipt_data['tax'] = price_match.group()
            
            elif 'total' in line.lower() and 'subtotal' not in line.lower():
                price_match = re.search(price_pattern, line)
                if price_match:
                    receipt_data['total'] = price_match.group()
            
            # Skip lines that are just weight measurements without product names
            elif re.search(weight_pattern, line) and not re.search(r'[A-Z]{2,}', line):
                # This is a standalone weight line, skip it
                i += 1
                continue
            
            # Skip lines that are just bulk pricing without product names
            elif re.search(bulk_pricing_pattern, line) and not re.search(r'[A-Z]{2,}', line):
                # This is a standalone bulk pricing line, skip it
                i += 1
                continue
            
            # Handle product lines with codes (like CAULIFLOWER007315012941F 3.12)
            elif re.search(product_code_pattern, line):
                # Extract product name (everything before the numeric code)
                product_match = re.match(r'([A-Z\s]+?)\d{12}', line)
                if product_match:
                    product_name = product_match.group(1).strip()
                    
                    # Extract price from the line
                    price_match = re.search(price_pattern, line)
                    if price_match:
                        price = price_match.group()
                        receipt_data['items'].append(f"{product_name} - {price}")
                    else:
                        # If no price on this line, check if next line has weight or bulk pricing info
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if re.search(weight_pattern, next_line) or re.search(bulk_pricing_pattern, next_line):
                                weight_price_match = re.search(price_pattern, next_line)
                                if weight_price_match:
                                    price = weight_price_match.group()
                                    receipt_data['items'].append(f"{product_name} - {price}")
                                    i += 1  # Skip the weight/bulk pricing line
                
            # Handle other items with prices (non-product-code format)
            elif (re.search(price_pattern, line) and 
                  not any(keyword in line.lower() for keyword in ['total', 'tax', 'subtotal', 'change', 'tender', 'credit']) and
                  not re.search(weight_pattern, line) and
                  not re.search(bulk_pricing_pattern, line) and
                  not re.search(r'^\d+\.\d+lb', line.lower())):
                
                # Clean up the line by removing extra codes
                cleaned_line = re.sub(r'\d{12}\s*F', '', line)  # Remove product codes
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()  # Clean whitespace
                
                if cleaned_line and len(cleaned_line) > 4:  # Only add if meaningful content remains
                    receipt_data['items'].append(cleaned_line)
            
            i += 1
        
        # Add Walmart identification
        walmart_data = self.identify_walmart_receipt(text)
        receipt_data['walmart_data'] = walmart_data
        
        # If it's Walmart, get Walmart-specific data
        if walmart_data['is_walmart']:
            walmart_specific = self.parse_walmart_specific_data(text)
            receipt_data['walmart_specific'] = walmart_specific
        
        return receipt_data
    
    def process_receipt(self, image_bytes: bytes) -> Dict:
        """
        Complete receipt processing pipeline
        """
        # Extract raw text
        raw_text = self.extract_text_basic(image_bytes)
        
        # Parse structured data
        structured_data = self.parse_receipt_structure(raw_text)
        
        # Add raw text to results
        structured_data['raw_text'] = raw_text
        
        return structured_data

def main():
    """
    Streamlit app for receipt text extraction
    """
    st.set_page_config(
        page_title="Receipt OCR Extractor",
        page_icon="🧾",
        layout="wide"
    )
    
    st.title("🧾 Walmart Receipt Detector & OCR Extractor")
    st.markdown("Upload a receipt image to check if it's from Walmart and extract structured data")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    show_raw_text = st.sidebar.checkbox("Show Raw Text", value=False)
    show_coordinates = st.sidebar.checkbox("Show Text Coordinates", value=False)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0, 100, 30)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a receipt image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a clear image of your receipt for best results"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(uploaded_file, caption="Receipt Image", use_column_width=True)
            
            # Show file details
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**File size:** {uploaded_file.size:,} bytes")
        
        with col2:
            st.subheader("🔍 Extraction Results")
            
            # Process image
            with st.spinner("Extracting text from receipt..."):
                extractor = ReceiptTextExtractor()
                image_bytes = uploaded_file.read()
                results = extractor.process_receipt(image_bytes)
            
            # Walmart identification section
            walmart_data = results.get('walmart_data', {})
            if walmart_data.get('is_walmart'):
                st.success(f"✅ **WALMART RECEIPT DETECTED** (Confidence: {walmart_data.get('confidence_score', 0)}%)")
                
                walmart_specific = results.get('walmart_specific', {})
                
                # Walmart-specific info
                if walmart_specific:
                    st.subheader("🏪 Walmart Store Information")
                    
                    # Create columns for Walmart data
                    walmart_col1, walmart_col2 = st.columns(2)
                    
                    with walmart_col1:
                        if walmart_specific.get('store_number'):
                            st.write(f"**Store #:** {walmart_specific['store_number']}")
                        if walmart_data.get('store_format'):
                            st.write(f"**Format:** {walmart_data['store_format']}")
                        if walmart_specific.get('terminal_number'):
                            st.write(f"**Terminal #:** {walmart_specific['terminal_number']}")
                    
                    with walmart_col2:
                        if walmart_specific.get('transaction_number'):
                            st.write(f"**Transaction #:** {walmart_specific['transaction_number']}")
                        if walmart_specific.get('operator_id'):
                            st.write(f"**Operator ID:** {walmart_specific['operator_id']}")
                        if walmart_specific.get('manager_name'):
                            st.write(f"**Manager:** {walmart_specific['manager_name']}")
                
                # Show Walmart indicators found
                if walmart_data.get('walmart_indicators_found'):
                    st.info(f"**Walmart indicators found:** {', '.join(walmart_data['walmart_indicators_found'])}")
            
            else:
                st.warning("❌ **NOT A WALMART RECEIPT**")
                st.info("This appears to be from a different retailer")
            
            st.divider()
            
            # Display general structured results
            if results['store_name']:
                st.success(f"**Store:** {results['store_name']}")
            
            # Create metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                if results['date']:
                    st.metric("Date", results['date'])
                else:
                    st.metric("Date", "Not found")
            
            with metrics_col2:
                if results['total']:
                    st.metric("Total", results['total'])
                else:
                    st.metric("Total", "Not found")
            
            with metrics_col3:
                st.metric("Items Found", len(results['items']))
            
            # Additional details
            if results['time']:
                st.write(f"**Time:** {results['time']}")
            if results['phone']:
                st.write(f"**Phone:** {results['phone']}")
            if results['subtotal']:
                st.write(f"**Subtotal:** {results['subtotal']}")
            if results['tax']:
                st.write(f"**Tax:** {results['tax']}")
        
        # Items section (full width)
        if results['items']:
            st.subheader("🛒 Items")
            
            # Create a nice table for items
            items_data = []
            for i, item in enumerate(results['items'], 1):
                items_data.append({"#": i, "Item": item})
            
            st.table(items_data)
        
        # Optional sections based on sidebar settings
        if show_coordinates:
            st.subheader("📍 Text with Coordinates")
            coord_data = extractor.extract_text_with_coordinates(image_bytes)
            
            if coord_data:
                # Filter by confidence threshold
                filtered_data = [item for item in coord_data if item['confidence'] >= confidence_threshold]
                
                st.write(f"Showing {len(filtered_data)} text elements with confidence ≥ {confidence_threshold}%")
                
                coord_df = []
                for item in filtered_data:
                    coord_df.append({
                        "Text": item['text'],
                        "Confidence": f"{item['confidence']}%",
                        "X": item['x'],
                        "Y": item['y'],
                        "Width": item['width'],
                        "Height": item['height']
                    })
                
                st.dataframe(coord_df, use_container_width=True)
        
        if show_raw_text:
            st.subheader("📝 Raw Extracted Text")
            if results['raw_text']:
                st.text_area("Raw Text", results['raw_text'], height=200)
            else:
                st.warning("No text was extracted from the image")
        
        # Download button for results
        st.subheader("💾 Download Results")
        
        # Prepare download content
        walmart_info = ""
        if results.get('walmart_data', {}).get('is_walmart'):
            walmart_specific = results.get('walmart_specific', {})
            walmart_info = f"""
WALMART RECEIPT DETAILS
======================
Confidence: {results['walmart_data'].get('confidence_score', 0)}%
Store Format: {results['walmart_data'].get('store_format', 'N/A')}
Store Number: {walmart_specific.get('store_number', 'N/A')}
Terminal Number: {walmart_specific.get('terminal_number', 'N/A')}
Transaction Number: {walmart_specific.get('transaction_number', 'N/A')}
Operator ID: {walmart_specific.get('operator_id', 'N/A')}
Manager: {walmart_specific.get('manager_name', 'N/A')}
Indicators Found: {', '.join(results['walmart_data'].get('walmart_indicators_found', []))}

"""
        
        download_content = f"""Receipt Extraction Results
=============================
IS WALMART RECEIPT: {results.get('walmart_data', {}).get('is_walmart', False)}
{walmart_info}
GENERAL INFORMATION
==================
Store: {results.get('store_name', 'N/A')}
Date: {results.get('date', 'N/A')}
Time: {results.get('time', 'N/A')}
Phone: {results.get('phone', 'N/A')}
Subtotal: {results.get('subtotal', 'N/A')}
Tax: {results.get('tax', 'N/A')}
Total: {results.get('total', 'N/A')}

Items:
{chr(10).join(f"- {item}" for item in results.get('items', []))}

Raw Text:
{results.get('raw_text', '')}
"""
        
        st.download_button(
            label="Download Results as Text File",
            data=download_content,
            file_name=f"receipt_extraction_{uploaded_file.name}.txt",
            mime="text/plain"
        )
    
    else:
        st.info("👆 Please upload a receipt image to get started")
        
        # Show example/instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Upload** a clear image of your receipt
        2. **Wait** for the OCR processing to complete
        3. **Review** the extracted information
        4. **Download** the results if needed
        
        **Tips for best results:**
        - Use good lighting when taking the photo
        - Ensure the receipt is flat and unfolded
        - Keep the camera steady to avoid blur
        - Make sure all text is visible and not cut off
        """)

if __name__ == "__main__":
    main()