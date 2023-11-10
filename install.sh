pip install gdown
gdown [INSERT URL]


download_url="https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar"
output_file="mistral-7B-v0.1.tar"

# Download the file
echo "Downloading file..."
wget "$download_url" -O "$output_file"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download finished successfully."

  # Untar the file
  echo "Untarring file..."
  tar -xf "$output_file"

  # Clean up the tar file
  rm "$output_file"
  
  echo "Extraction completed."
else
  echo "Download failed. Please try again."
fi
