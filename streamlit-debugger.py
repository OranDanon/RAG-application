import threading
import streamlit
import streamlit.web.bootstrap
import sys
import os



def run_streamlit():
    # Get the absolute path to your streamlit app
    streamlit_app_path = os.path.abspath("chatbot.py")

    # Set up command line arguments
    sys.argv = ["streamlit", "run", streamlit_app_path, "--server.port", "8501"]

    # Use the correct bootstrap method with required arguments
    streamlit.web.bootstrap.run(
        main_script_path=streamlit_app_path,
        is_hello=False,
        args=[],
        flag_options={}
    )


if __name__ == "__main__":
    # # Start FastAPI in a separate thread
    # fastapi_thread = threading.Thread(target=run_fastapi)
    # fastapi_thread.daemon = True
    # fastapi_thread.start()

    # Run Streamlit in the main thread
    run_streamlit()