**Prerequisites:**

1.  **VS Code:** You need VS Code installed on your local machine.
2.  **VS Code Extensions:** Install the official `Python` and `Jupyter` extensions from the VS Code Marketplace.
3.  **Local Python Environment:** You need a Python installation on your local machine.
4.  **Install `jupyter_http_over_ws` locally:** This package enables the connection. Open your local terminal or command prompt and run:
    ```bash
    pip install jupyter_http_over_ws
    jupyter serverextension enable --py jupyter_http_over_ws
    ```

**Steps:**

1.  **Start a Colab Runtime:**
    * Go to [Google Colab](https://colab.research.google.com/).
    * Create a new notebook or open an existing one (the content of *this* notebook doesn't matter much initially, you just need the runtime).
    * Go to `Runtime` -> `Change runtime type`.
    * Select `GPU` (or `TPU` if needed) as the hardware accelerator.
    * Click `Save`.

2.  **Get the Connection URL from Colab:**
    * In a code cell in the Colab notebook you just opened/created, run the following code. This will install the necessary package *in the Colab environment* and then print an authentication URL:
        ```python
        !pip install jupyter_http_over_ws
        !jupyter serverextension enable --py jupyter_http_over_ws

        # Run this cell to connect to the Colab runtime from VS Code
        # It will print a URL that you need to copy
        from google.colab.output import eval_js
        print(eval_js("google.colab.kernel.proxyPort(8888, {'cache': false})"))
        ```
    * Run this cell. It will output a URL that looks something like `https://colab.research.google.com/tun/m/gpu-xxxxxxxxxx-xxxx/`. **Copy this entire URL.** You'll need it in VS Code. Keep this Colab tab open.

3.  **Connect VS Code to the Colab Runtime:**
    * Open VS Code on your local machine.
    * Open the local `.ipynb` notebook file you want to run on the Colab GPU.
    * Click the `Select Kernel` button in the top-right corner of the notebook editor (it might currently say "Python 3.x.x" or similar).
    * From the dropdown menu, choose `Existing Jupyter Server...`.
    * You'll be prompted to `Enter the URL of the running Jupyter server`. **Paste the URL you copied from the Colab output** in the previous step and press Enter.
    * You might be asked for a name for the server (optional, just helps you identify it later).

4.  **Run Your Notebook:**
    * VS Code should now connect. The kernel status in the top-right corner should update to indicate it's connected to the remote server (often showing a generic Python version or "Busy" briefly).
    * You can now run cells in your *local* notebook file. The execution will happen on the Google Colab runtime using its GPU. Any output, plots, or errors will appear directly in your VS Code interface.

**Important Considerations:**

* **Data Files:** While the `.ipynb` notebook file itself remains local, any data files (CSVs, images, model weights, etc.) that your code needs *must* be accessible to the *Colab runtime*. You still need to upload them to the Colab environment. Common methods include:
    * Using `google.colab.files.upload()` within a notebook cell run from VS Code.
    * Mounting your Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) and accessing files from there.
    * Cloning a Git repository (`!git clone ...`).
    * Downloading files using `!wget` or Python libraries like `requests`.
* **Runtime Disconnection:** Colab runtimes have idle timeouts (typically 90 minutes) and absolute timeouts (usually 12 hours). If the Colab runtime disconnects, your VS Code connection will break, and you'll need to restart the runtime in the Colab browser tab, re-run the connection code cell to get a *new* URL, and reconnect VS Code.
* **Package Installation:** Any Python packages your notebook requires must be installed *in the Colab environment*. You can do this by running `!pip install <package_name>` in a cell from VS Code once connected.
