"""
Invoice Processing Script
=========================
This script processes PDF files and images to extract invoice data using Azure OpenAI.
It converts the extracted information into structured CSV format with both fixed and dynamic schemas.

Main functionality:
- Extracts text from PDF files using pdfplumber
- Processes images by converting them to base64 format
- Uses Azure OpenAI to analyze and structure the data
- Outputs results in two CSV formats: fixed schema and dynamic schema
"""

# Standard library imports
import os           # For file and directory operations
import csv          # For CSV file creation and manipulation
import uuid         # For generating unique identifiers for output files
import base64       # For encoding images to base64 format

# Third-party imports
import pdfplumber   # For extracting text from PDF files
from PIL import Image  # For image processing (imported but not currently used)

# Local imports
import AppConfig.Confighelper as ConHP  # Configuration helper for API settings
from openai import AzureOpenAI          # Azure OpenAI client for LLM calls


# Configure Azure OpenAI client with API credentials from config
os.environ['AZURE_OPENAI_API_KEY'] = ConHP.get_config('api_key')
client = AzureOpenAI(
    api_version=ConHP.get_config('api_version'),
    azure_endpoint=ConHP.get_config('api_base')
)
# Get the deployment name for the generic model from configuration
generic_model = ConHP.get_config('azure_generic_deployment_name')




#-------------------------------------- gemini setup -----------------------------------------

import google.generativeai as genai  # Google Gemini AI
genai.configure(api_key=ConHP.get_config('Gemini_API_KEY'))

def llm_call_gemini(prompt, data=None, base64_img=None):
    """
    Makes a call to Google Gemini for vision and text processing.
    """
    try:
        # Use the latest Gemini 2.5 Flash model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare content
        content = []
        
        # Add text content
        full_prompt = prompt
        if data:
            full_prompt = f"{prompt}\n\n{data}"
        content.append(full_prompt)
        
        # Add image if provided
        if base64_img:
            import io
            from PIL import Image as PILImage
            
            # Convert base64 to PIL Image
            image_data = base64.b64decode(base64_img)
            image = PILImage.open(io.BytesIO(image_data))
            content.append(image)
        
        # Generate response
        response = model.generate_content(content)
        return response.text
    
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return None

#-------------------------------------- gemini setup end -----------------------------------------






def llm_call(prompt, data=None, base64_img=None):
    """
    Makes a call to Azure OpenAI to process text and/or image data.
    
    Args:
        prompt (str): The instruction prompt to send to the LLM
        data (str, optional): Text data to be analyzed (e.g., extracted PDF text)
        base64_img (str, optional): Base64 encoded image data for vision analysis
    
    Returns:
        str: The LLM's response content, or None if an error occurs
    """
    try:
        # Initialize content with the text prompt
        content = [{"type": "text", "text": prompt}]
        
        # If additional text data is provided, append it to the prompt
        if data:
            content[0]["text"] = f"{prompt}\n\n{data}"
        
        # If an image is provided, add it to the content for vision analysis
        if base64_img:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}"
                }
            })

        # Make the API call to Azure OpenAI
        response = client.chat.completions.create(
            model=generic_model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=4000  # Limit response length
        )

        # Extract and return the response content
        return response.choices[0].message.content

    except Exception as e:
        print(f"\nError during LLM call: {e}\n")
        return None

def save_csv_response(response, file_path, file_type="file"):
    """
    Saves the LLM response as two separate CSV files (fixed and dynamic schema).
    
    Args:
        response (str): The LLM response containing CSV data separated by '=== CSV 2 ==='
        file_path (str): Path of the original file being processed (for naming output files)
        file_type (str): Type of file being processed ("pdf" or "image")
    
    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        # Split the response into two CSV sections using the delimiter
        parts = response.split("=== CSV 2 ===")
        if len(parts) != 2:
            raise ValueError("Response does not contain expected '=== CSV 2 ===' delimiter.")

        # Extract and clean the two CSV sections
        csv1_str = parts[0].strip()  # Fixed schema CSV
        csv2_str = parts[1].strip()  # Dynamic schema CSV

        # Parse CSV strings into list of lists for writing
        csv1_data = [row for row in csv.reader(csv1_str.splitlines())]
        csv2_data = [row for row in csv.reader(csv2_str.splitlines())]

        # Create output directory if it doesn't exist
        output_folder = os.path.join(os.getcwd(), "Outputs")
        os.makedirs(output_folder, exist_ok=True)

        # Generate unique filenames based on original file name
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        unique_id = uuid.uuid4()  # Prevent filename conflicts
        output_path_csv1 = os.path.join(output_folder, f"{base_filename}_fixed_schema_{unique_id}.csv")
        output_path_csv2 = os.path.join(output_folder, f"{base_filename}_dynamic_schema_{unique_id}.csv")

        # Write the first CSV file (fixed schema)
        with open(output_path_csv1, mode='w', newline='', encoding='utf-8') as file1:
            writer = csv.writer(file1)
            writer.writerows(csv1_data)

        # Write the second CSV file (dynamic schema)
        with open(output_path_csv2, mode='w', newline='', encoding='utf-8') as file2:
            writer = csv.writer(file2)
            writer.writerows(csv2_data)

        # Confirm successful file creation
        print(f"\nCSV 1 (fixed schema) saved to {output_path_csv1}")
        print(f"CSV 2 (dynamic schema) saved to {output_path_csv2}\n")
        return True

    except Exception as e:
        print(f"\nResponse is not in valid two-CSV format: {e}\n")
        return False


def get_pdf_files(folder_path):
    """
    Generator function that yields PDF file paths from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
    
    Yields:
        str: Full path to each PDF file found in the folder
    """
    # Validate that the folder exists
    if not os.path.isdir(folder_path):
        print(f"\nInvalid folder path: {folder_path}\n")
        return

    # Find all PDF files in the folder (case-insensitive)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    # Check if any PDF files were found
    if not pdf_files:
        print("\nNo PDF files found in the folder.\n")
        return

    # Yield the full path for each PDF file
    for pdf_file in pdf_files:
        yield os.path.join(folder_path, pdf_file)

def process_single_pdf(pdf_path):
    """
    Processes a single PDF file by extracting text and sending it to the LLM.
    
    Args:
        pdf_path (str): Full path to the PDF file to be processed
    """
    print(f"\nProcessing PDF: {pdf_path}\n")
    try:
        # Open and extract text from all pages of the PDF
        with pdfplumber.open(pdf_path) as pdf:
            # Concatenate text from all pages
            full_text = ''.join(page.extract_text() for page in pdf.pages)
            print(f"\nExtracted text from {pdf_path}\n")

            # Get the PDF processing prompt from configuration
            prompt = ConHP.get_config('prompt_pdf')

            # Send the extracted text to the LLM for analysis
            response = llm_call_gemini(prompt, data=full_text)  
            
            # Save the response if we received one
            if response:  
                try:
                    csv_data = [row for row in csv.reader(response.splitlines())]
                    
                    output_folder = os.path.join(os.getcwd(), "Outputs")
                    os.makedirs(output_folder, exist_ok=True)
                    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                    unique_filename = f"{base_filename}_pdf_{uuid.uuid4()}.csv"
                    output_path = os.path.join(output_folder, unique_filename)

                    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerows(csv_data)
                    
                    print(f"\nCSV saved to {output_path}\n")
                except Exception as e:
                    print(f"\nResponse is not in valid CSV format: {e}\n")
            else:  
                print("\nNo response received from LLM.\n")

    except Exception as e:
        print(f"\nError processing {pdf_path}: {e}\n")

def process_pdf(folder_path):
    """
    Processes all PDF files in a given folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF files to process
    """
    # Process each PDF file found in the folder
    for pdf_path in get_pdf_files(folder_path):
        process_single_pdf(pdf_path)


def get_image_files(folder_path):
    """
    Generator function that yields image file paths from a given folder.
    
    Args:
        folder_path (str): Path to the folder containing image files
    
    Yields:
        str: Full path to each image file found in the folder
    """
    # Validate that the folder exists
    if not os.path.isdir(folder_path):
        print(f"\nInvalid folder path: {folder_path}\n")
        return

    # Define supported image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    # Find all image files in the folder (case-insensitive)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    # Check if any image files were found
    if not image_files:
        print("\nNo image files found in the folder.\n")
        return

    # Yield the full path for each image file
    for image_file in image_files:
        yield os.path.join(folder_path, image_file)

def process_single_image(image_path):
    """
    Processes a single image file by converting it to base64 and sending it to the LLM.
    
    Args:
        image_path (str): Full path to the image file to be processed
    """
    print(f"\nProcessing Image: {image_path}\n")
    try:
        # Read the image file and convert it to base64 encoding
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            print(f"\nConverted image to Base64: {image_path}\n")

            # Get the image processing prompt from configuration
            prompt = ConHP.get_config('prompt_image')

            # Send the base64 image to the LLM for vision analysis
            response = llm_call_gemini(prompt, base64_img=base64_image)

            # Save the response if we received one
            if response:  
                #save_csv_response(response, image_path, file_type="image")
                try:
                    csv_data = [row for row in csv.reader(response.splitlines())]
                    
                    output_folder = os.path.join(os.getcwd(), "Outputs")
                    os.makedirs(output_folder, exist_ok=True)
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    unique_filename = f"{base_filename}_image_{uuid.uuid4()}.csv"
                    output_path = os.path.join(output_folder, unique_filename)

                    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerows(csv_data)
                    
                    print(f"\nCSV saved to {output_path}\n")
                except Exception as e:
                    print(f"\nResponse is not in valid CSV format: {e}\n")
            else:  
                print("\nNo response received from LLM.\n")

    except Exception as e:
        print(f"\nError processing {image_path}: {e}\n")

def process_images(folder_path):
    """
    Processes all image files in a given folder.
    
    Args:
        folder_path (str): Path to the folder containing image files to process
    """
    # Process each image file found in the folder
    for image_path in get_image_files(folder_path):
        process_single_image(image_path)


# Main execution block
if __name__ == "__main__":
    # Define the folder containing files to be processed
    folder_path = "Data"
    
    # Process all PDF files in the folder
    process_pdf(folder_path)
    
    # Process all image files in the folder
    process_images(folder_path)