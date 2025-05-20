import streamlit as st
import os
import glob

class Sidebar:
    """
    Class to handle the sidebar components of the Streamlit application
    """
    
    @staticmethod
    def create_sidebar():
        """Create the sidebar with navigation and data selection options"""
        with st.sidebar:
            st.title("Livestock Analyzer")
            
            # Navigation
            st.header("Navigation")
            page = st.radio(
                "Go to",
                options=["Home", "Data Explorer", "Analysis", "Forecasting", "About"]
            )
            
            # Data selection (only show if on relevant pages)
            if page in ["Data Explorer", "Analysis", "Forecasting"]:
                st.header("Data Selection")
                
                # Get available datasets
                datasets = Sidebar._get_available_datasets()
                
                if datasets:
                    selected_dataset = st.selectbox(
                        "Select dataset",
                        options=datasets
                    )
                else:
                    st.warning("No datasets found. Please upload a dataset first.")
                    selected_dataset = None
                
                # File upload option
                st.header("Upload New Data")
                uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
                
                if uploaded_file is not None:
                    # Save the uploaded file
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Uploaded {uploaded_file.name} successfully!")
                    
                    # Add to available datasets
                    if uploaded_file.name not in datasets:
                        datasets.append(uploaded_file.name)
                        selected_dataset = uploaded_file.name
            
            return {
                "page": page,
                "selected_dataset": selected_dataset if page in ["Data Explorer", "Analysis", "Forecasting"] else None,
                "uploaded_file": uploaded_file if 'uploaded_file' in locals() else None
            }
    
    @staticmethod
    def _get_available_datasets():
        """Get a list of available datasets in the data directory"""
        # Check if data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Get all CSV files in the data directory
        csv_files = glob.glob("data/*.csv")
        
        # Extract file names
        file_names = [os.path.basename(file) for file in csv_files]
        
        return file_names
