## Multifunctional Browser Extension for Accessibility

### **Setting up Django**

1.  Extract all the files into one folder.

2.  Create a Virtual Environment to install dependencies:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  Install all dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This includes the `pytesseract` library for OCR.)*

4.  **Install and Set up Tesseract OCR Engine:**
    pytesseract is a Python wrapper for the Tesseract OCR engine. You need to install the Tesseract executable separately on your system. Follow the instructions for your operating system:

    * **Windows:**
        Download the installer from the [Tesseract-OCR GitHub Releases page](https://github.com/UB-Mannheim/tesseract/wiki). Run the installer and follow the prompts. Make sure to check the option to "Add Tesseract to the system PATH" during installation, or manually add the installation directory to your system's PATH environment variable.


    After installation, ensure the `tesseract` command is available in your system's PATH. You can test this by opening a new terminal or command prompt and typing `tesseract --version`.

5.  Set your `GROQ_API_KEY` from groqcloud, `HF_TOKEN` from huggingface, and `GOOGLE_API_KEY` from Google Cloud in a `.env` file.

6.  Move to the main app folder and perform migrations:
    ```bash
    cd inText
    python manage.py makemigrations
    python manage.py migrate
    ```

7.  Run the server:
    ```bash
    python manage.py runserver
    ```

### **Setting up the Extension in Your Browser**

1.  Open the browser and navigate to the extensions page.

2.  Click on the "Load unpacked" button.

3.  Navigate to the `sidebar` folder and select it.

4.  Perform the required actions from the extension interface.

### **Teammates**

* Kamalesh N
* Idris Malik
* Shashank Gattu
* Vasudeva Reddy
