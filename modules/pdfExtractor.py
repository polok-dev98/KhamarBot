import os
import json
import glob
from pathlib import Path
import pdfplumber
from tqdm import tqdm
import re

# Configuration
PDF_DATA_DIR = "./pdf_data"
JSON_OUTPUT_DIR = "./pdf_file"

def clean_text_content(text):
    """
    Clean text by removing special characters and unwanted spaces
    while preserving Bengali and basic English punctuation
    """
    if not text:
        return text
    
    # Keep Bengali characters, English letters, numbers, and basic punctuation
    # Bengali Unicode range: \u0980-\u09FF
    # Basic punctuation: . , ! ? ; : - ( ) [ ] { } ' " ... । (Bengali danda)
    allowed_chars_pattern = r'[^\u0980-\u09FFa-zA-Z0-9\s\.\,\!\?\;\:\-\_\(\)\[\]\{\}\'\""।…]'
    
    # Remove special characters
    cleaned_text = re.sub(allowed_chars_pattern, '', text)
    
    # Remove extra whitespaces (multiple spaces, tabs, newlines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Clean up space around punctuation
    cleaned_text = re.sub(r'\s+([।.,!?;:])', r'\1', cleaned_text)  # Remove space before punctuation
    cleaned_text = re.sub(r'([।.,!?;:])\s+', r'\1 ', cleaned_text)  # Ensure space after punctuation
    
    # Remove spaces at the beginning and end
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(PDF_DATA_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

def extract_text_with_pdfplumber(pdf_path):
    """Extract text from PDF using pdfplumber"""
    try:
        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    cleaned_text = clean_extracted_text(text)
                    text_content.append({
                        'page': page_num,
                        'text': cleaned_text
                    })
        return text_content
    except Exception as e:
        print(f"Error with pdfplumber for {pdf_path}: {e}")
        return []

def clean_extracted_text(text):
    """Clean and normalize extracted text"""
    # Remove extra whitespaces and normalize newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Apply additional cleaning to remove special characters
    text = clean_text_content(text)
    
    return text

def detect_language(text):
    """Detect if text is primarily Bengali, English, or mixed"""
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total_chars = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text)) or 1
    
    bengali_ratio = bengali_chars / total_chars
    english_ratio = english_chars / total_chars
    
    if bengali_ratio > 0.6:
        return "bengali"
    elif english_ratio > 0.6:
        return "english"
    else:
        return "mixed"

def is_content_rich(text, min_length=50):
    """Check if text has substantial content (not just headers/toc)"""
    if len(text.strip()) < min_length:
        return False
    
    # Check if it's likely a table of contents or page number
    toc_patterns = [
        r'contents|table of contents|index|সূচিপত্র|মুখবন্ধ',
        r'page\s+\d+|\d+\s+of\s+\d+',
        r'^\d+$'  # Just page numbers
    ]
    
    for pattern in toc_patterns:
        if re.search(pattern, text.lower()):
            return False
    
    return True

def split_into_sections(text, max_length=800):
    """Split text into logical sections for both Bengali and English"""
    sections = []
    
    # Split by paragraphs (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_section = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph or not is_content_rich(paragraph):
            continue
            
        # If adding this paragraph would exceed max length and we have content, start new section
        if len(current_section) + len(paragraph) > max_length and current_section:
            sections.append(current_section)
            current_section = paragraph
        else:
            if current_section:
                current_section += "\n\n" + paragraph
            else:
                current_section = paragraph
    
    # Add the last section
    if current_section and is_content_rich(current_section):
        sections.append(current_section)
    
    # If no sections were created but we have content, split by sentences
    if not sections and text.strip() and is_content_rich(text):
        # Different sentence endings for Bengali and English
        sentence_endings = r'[।?!\.]\s+'
        sentences = re.split(sentence_endings, text)
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or not is_content_rich(sentence, min_length=20):
                continue
                
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                sections.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    # Add appropriate sentence ending
                    lang = detect_language(current_chunk)
                    if lang == "bengali":
                        current_chunk += "। " + sentence
                    else:
                        current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk and is_content_rich(current_chunk):
            sections.append(current_chunk)
    
    return sections

def process_pdf_to_json(pdf_path):
    """Convert PDF to JSON format - handles both Bengali and English"""
    pdf_name = Path(pdf_path).stem
    output_json_path = os.path.join(JSON_OUTPUT_DIR, f"{pdf_name}.json")
    
    print(f"Processing: {pdf_name}")
    
    # Extract text from PDF using pdfplumber
    pdf_content = extract_text_with_pdfplumber(pdf_path)
    
    if not pdf_content:
        print(f"  ⚠️  No content extracted from {pdf_name}")
        return
    
    json_data = []
    total_pages = len(pdf_content)
    bengali_pages = 0
    english_pages = 0
    mixed_pages = 0
    
    for page_data in pdf_content:
        page_num = page_data['page']
        page_text = page_data['text']
        
        # Detect language of the page
        page_language = detect_language(page_text)
        
        if page_language == "bengali":
            bengali_pages += 1
        elif page_language == "english":
            english_pages += 1
        else:
            mixed_pages += 1
        
        # Skip pages without substantial content
        if not is_content_rich(page_text):
            continue
        
        # Split into sections
        sections = split_into_sections(page_text)
        
        for section in sections:
            if not section.strip() or not is_content_rich(section):
                continue
            
            # Apply final cleaning to the section content
            cleaned_section = clean_text_content(section)
            
            # Skip if content is empty after cleaning
            if not cleaned_section.strip():
                continue
            
            # Create JSON entry with null header
            entry = {
                "content": cleaned_section,
                "header": None,  # Header set to null as requested
                "page": page_num
            }
            
            json_data.append(entry)
    
    # Print language statistics
    print(f"  📊 Language analysis: {bengali_pages} Bengali, {english_pages} English, {mixed_pages} mixed pages")
    print(f"  ✅ Converted to JSON: {len(json_data)} entries")
    
    # Save as JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    return len(json_data)

def main():
    """Main function to process all PDFs in pdf_data directory"""
    print("📚 PDF to JSON Converter (Handles Both Bengali & English)")
    print("=" * 60)
    
    ensure_directories()
    
    pdf_files = glob.glob(os.path.join(PDF_DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{PDF_DATA_DIR}' directory")
        print(f"   Please place your PDF files in the '{PDF_DATA_DIR}' folder")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to process...")
    
    total_entries = 0
    successful_conversions = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            entries_count = process_pdf_to_json(pdf_file)
            if entries_count:
                total_entries += entries_count
                successful_conversions += 1
            else:
                print(f"  ⚠️  No valid content extracted from {Path(pdf_file).stem}")
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Conversion Summary:")
    print(f"   ✅ Successfully processed: {successful_conversions}/{len(pdf_files)} PDF files")
    print(f"   📝 Total JSON entries created: {total_entries}")
    print(f"   💾 JSON files saved to: '{JSON_OUTPUT_DIR}/'")
    
    if successful_conversions > 0:
        print("\n🎉 PDF processing completed successfully!")
    else:
        print("\n❌ No PDF files were successfully processed. Please check:")
        print("   - PDF files are not password protected")
        print("   - PDF files contain extractable text (not scanned images)")
        print("   - PDF files are in the correct directory")

if __name__ == "__main__":
    main()