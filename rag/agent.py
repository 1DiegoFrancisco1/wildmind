import os
import re
import boto3
import requests

BUCKET_NAME = "wildmind-docs"
LOCAL_DOCS_PATH = "docs/"

# Curated list of WWF wildlife PDFs
# Note: Direct URLs used due to Cloudflare protection on the publications index page
WWF_PDFS = [
    {
        "url": "https://www.worldwildlife.org/documents/695/5bgjf8vbiy_WWF_Africas_Forgotten_Fishes.pdf",
        "filename": "WWF_Africas_Forgotten_Fishes.pdf"
    },
    {
        "url": "https://www.worldwildlife.org/documents/463/2i9pn6b82s_Climate_Crowd_Climate_Change_and_HWC_Final.pdf",
        "filename": "WWF_Climate_Change_and_Human_Wildlife_Conflict.pdf"
    },
    {
        "url": "https://www.worldwildlife.org/documents/738/7f9g41e87d_Guidance_for_Linear_Infrastructure_2025June26_FINAL.pdf",
        "filename": "WWF_Snow_Leopard_Linear_Infrastructure.pdf"
    },
    {
        "url": "https://www.worldwildlife.org/documents/2281/FINAL_26_3202_UN_CMS_Migratory_Fish_Report_v6c_031826_uL4zPT8.pdf",
        "filename": "WWF_Global_Assessment_Migratory_Freshwater_Fishes.pdf"
    },
    {
        "url": "https://www.worldwildlife.org/documents/2277/v8_legislation-report_2025_hr-pages.pdf",
        "filename": "WWF_Law_of_the_Tiger.pdf"
    },
    {
        "url": "https://www.worldwildlife.org/documents/2382/WWF_Legacy_Report_2026_PressReady.pdf",
        "filename": "WWF_Legacy_Report_2026.pdf"
    },
    {
        "url": "https://www.worldwildlife.org/documents/2379/Primates_for_Purchase_Report.pdf",
        "filename": "WWF_Primates_for_Purchase.pdf"
    }
]


def download_pdf(url: str, filename: str):
    """Downloads a PDF from a URL to the local docs folder."""
    local_path = os.path.join(LOCAL_DOCS_PATH, filename)

    # Skip if already downloaded
    if os.path.exists(local_path):
        print(f"{filename} already exists, skipping...")
        return local_path

    print(f"Downloading {filename}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=30)
    
    # Check if download was successful
    if response.status_code != 200:
        print(f"Failed to download {filename} — status {response.status_code}")
        return None

    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"✅ Saved {filename}!")
    return local_path


def upload_to_s3(local_path: str, filename: str):
    """Uploads a local file to S3."""
    s3 = boto3.client("s3", region_name="us-east-2")

    print(f"Uploading {filename} to S3...")
    s3.upload_file(local_path, BUCKET_NAME, filename)
    print(f"✅ {filename} uploaded to S3!")


def run_agent():
    """Main agent — downloads WWF PDFs and uploads them to S3."""
    print("🤖 WildMind Agent starting...")
    print(f"Processing {len(WWF_PDFS)} documents...\n")

    successful = 0
    for pdf in WWF_PDFS:
        print(f"--- {pdf['filename']} ---")
        
        # Download locally
        local_path = download_pdf(pdf["url"], pdf["filename"])
        
        if local_path:
            # Upload to S3
            upload_to_s3(local_path, pdf["filename"])
            successful += 1
        
        print()

    print(f"🎉 Agent finished! {successful}/{len(WWF_PDFS)} documents ready in S3.")


if __name__ == "__main__":
    run_agent()