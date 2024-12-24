
import ollama
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import StringIO
from methods import *
from transformers import pipeline
from langchain_ollama import ChatOllama
from functools import lru_cache


st.set_page_config(layout="wide")

# --- Data Quality Analysis Section ---
def data_quality_analysis():

    # Initialize session state keys if they don't exist
    session_state_defaults = {
        'data': None,
        'previous_file_name': None,
        'show_data': False,
        'describe_data': False,
        'missing_analysis_run': False,
        'missing_values_handled': False,
        'duplicates_handled': False,
        'outlier_analysis_run': False,
        'outliers_handled': False,
        'visualize_data_run': False,
        'correlation_run': False,
        'type_converted': False,
        'columns_renamed': False,
        'data_type_analysis_clicked': False,
        'columns_dropped': False,
    }
    for key, value in session_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.sidebar.title("📊 Data Quality Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"], key='file_uploader')

    if uploaded_file is not None:
        # Reset the session state when a new file is uploaded
        if st.session_state['previous_file_name'] != uploaded_file.name:
        # Clear only dataset-related session state
            for key in ['data', 'previous_file_name']:
                st.session_state[key] = session_state_defaults[key]
            st.session_state['previous_file_name'] = uploaded_file.name

        # Load the new dataset
        if st.session_state['data'] is None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    csv_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    df = pd.read_csv(csv_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                st.session_state['data'] = df
                st.sidebar.success("Dataset uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        else:
            df = st.session_state['data'].copy()
        
        
        if st.sidebar.button("Chat Using RAG") :
            chat_with_rag(df)

        #show data
        if st.sidebar.button("Show Data", key='show_data_btn'):
            reset_all_flags()
            st.session_state['show_data'] = True

        if 'show_data' in st.session_state and st.session_state['show_data']:
            reset_all_flags()
            st.header("Data")
            st.write(df.head())
        
        #describe data
        if st.sidebar.button("Describe Data", key='describe_data_btn'):
            reset_all_flags()
            st.session_state['describe_data'] = True

        if 'describe_data' in st.session_state and st.session_state['describe_data']:
            reset_all_flags()
            st.header("Data Description")
            st.table(describe_data(df))
        #data type analysis
        if st.sidebar.button("Data Type Analysis", key='data_type_btn'):
            reset_all_flags()
            st.session_state['data_type_analysis_clicked'] = True
            df = data_types_analysis(df)
        if 'type_converted' in st.session_state and st.session_state['type_converted']:
            st.write(df)
            st.session_state['type_converted'] = False
        
        #data type conversion
        # Add Before and After functionality for Data Type Conversion
        selected_column = st.sidebar.selectbox("Select a column to convert", df.columns, key="convert_col")
        new_type = st.sidebar.selectbox("Select the new data type", ["int", "float", "str", "datetime"], key="new_type")
        
        if st.sidebar.button("Preview Data Type Conversion", key='preview_convert_btn'):
            preview_df = df.copy()
            try:
                if new_type == "datetime":
                    preview_df[selected_column] = pd.to_datetime(preview_df[selected_column], errors='coerce')
                elif new_type in ["int", "float"]:
                    preview_df[selected_column] = pd.to_numeric(preview_df[selected_column], errors='coerce')
                    if new_type == "int":
                        preview_df[selected_column] = preview_df[selected_column].astype(int)
                    elif new_type == "float":
                        preview_df[selected_column] = preview_df[selected_column].astype(float)
                else:
                    preview_df[selected_column] = preview_df[selected_column].astype(new_type)
                
                # Show Before and After the conversion
                st.write("Data Before Conversion:")
                st.write(df[selected_column].head())
                st.write("Data After Conversion:")
                st.write(preview_df[selected_column].head())
                
                st.session_state['type_converted'] = False
            except Exception as e:
                st.error(f"Error previewing conversion: {e}")

        if st.sidebar.button("Convert Data Type", key='convert_btn'):
            try:
                if new_type == "datetime":
                    df[selected_column] = pd.to_datetime(df[selected_column], errors='coerce')
                elif new_type in ["int", "float"]:
                    df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce')
                    if new_type == "int":
                        df[selected_column] = df[selected_column].fillna(0).astype(int)
                    elif new_type == "float":
                        df[selected_column] = df[selected_column].astype(float)
                else:
                    df[selected_column] = df[selected_column].astype(new_type)
                st.session_state['data'] = df
                st.success(f"Column '{selected_column}' converted to {new_type} successfully!")
                reset_all_flags()
            except Exception as e:
                st.error(f"Error converting column '{selected_column}': {e}")


        if st.sidebar.button("Column Name Analysis"):
            reset_all_flags()
            st.session_state['columns_renamed'] = True

        if 'columns_renamed' in st.session_state and st.session_state['columns_renamed']:
            df = column_names_analysis(df)
            st.session_state['data'] = df
            

        if 'columns_renamed' in st.session_state and st.session_state['columns_renamed']:
            reset_all_flags()
            st.session_state['columns_renamed'] = True

        if st.sidebar.button("Missing Value Analysis", key='missing_val_btn'):
            reset_all_flags()
            st.session_state['missing_analysis_run'] = True
        
        if 'missing_analysis_run' in st.session_state and st.session_state['missing_analysis_run']:
            st.header("Missing Value Analysis")
            missing_value_analysis(df)
            st.session_state['missing_analysis_run'] = False

        column = st.sidebar.selectbox("Select Column (optional)", df.columns, key="missing_col")
        method = st.sidebar.selectbox("Select Method", ["mean", "median", "mode", "drop"], key="missing_method")
        

        # Preview the effect before applying
        if st.sidebar.button("Preview Missing Value Handling", key='preview_missing_btn'):
            df_preview = handle_missing_values(df, method, column)
            st.subheader("Data Before Handling Missing Values")
            st.write(df)
            missing_value_analysis(df)
            
            st.subheader(f"Data After Applying '{method}' Method")
            st.write(df_preview)
            missing_value_analysis(df_preview)


        if st.sidebar.button("Handle Missing Values", key='handle_missing_btn'):
            reset_all_flags()
            df = handle_missing_values(df, method, column)
            st.session_state['data'] = df
            st.session_state['missing_values_handled'] = True


        if 'missing_values_handled' in st.session_state and st.session_state['missing_values_handled']:
            st.header("Data after Handling Missing Values")
            st.write(df)
            missing_value_analysis(df)
            st.session_state['missing_values_handled'] = False

        # Drop Columns Feature
        st.sidebar.subheader("Drop Columns")
        columns_to_drop = st.sidebar.multiselect("Select Columns to Drop", options=df.columns, key="columns_to_drop")
        
        if st.sidebar.button("Drop Selected Columns", key="drop_columns_btn"):
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                st.session_state['data'] = df
                st.success(f"Columns {columns_to_drop} have been dropped successfully!")
            else:
                st.warning("No columns selected for dropping.")

        if st.sidebar.button("Handle Duplicates", key='handle_duplicates_btn'):
            reset_all_flags()
            df = handle_duplicates(df)

        if 'duplicates_handled' in st.session_state and st.session_state['duplicates_handled']:
            st.header("Data after Handling Duplicates")
            st.write(df)
            st.session_state['duplicates_handled'] = False
# Button to apply the removal of duplicates
        if st.sidebar.button("Remove Duplicate Rows", key='remove_duplicates'):
            df.drop_duplicates(inplace=True)
            st.success("Duplicate rows removed.")
            st.session_state['data'] = df
            st.session_state['duplicates_handled'] = True

        column_for_outlier = st.sidebar.selectbox("Select Column for Outlier Analysis", df.select_dtypes(include=['float64', 'int64']).columns, key="outlier_col")

        if st.sidebar.button("Outlier Analysis", key='outlier_analysis_btn'):
            reset_all_flags()
            st.session_state['outlier_analysis_run'] = True
        
        
        if 'outlier_analysis_run' in st.session_state and st.session_state['outlier_analysis_run']:
            st.header("Outlier Analysis")
            lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)
            if lower_bound is not None and upper_bound is not None:
                outlier_method = st.sidebar.selectbox("Select Outlier Handling Method", ['clip', 'drop'], key="outlier_method")
                if st.sidebar.button("Preview Outlier Handling", key='preview_outliers_btn'):
                    # Visualize the data before handling
                    fig_before, ax_before = plt.subplots()
                    sns.boxplot(x=df[column_for_outlier], ax=ax_before)
                    st.subheader("Before Handling Outliers")
                    st.pyplot(fig_before)
                    
                    # Handle the outliers and preview the result
                    df_preview = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)
                    
                    # Visualize the data after handling
                    fig_after, ax_after = plt.subplots()
                    sns.boxplot(x=df_preview[column_for_outlier], ax=ax_after)
                    st.subheader("After Handling Outliers")
                    st.pyplot(fig_after)
                    st.write("Outliers will be handled using the selected method.")

                if st.sidebar.button("Handle Outliers", key='handle_outliers_btn'):
                    df = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)
                    st.session_state['data'] = df
                    st.session_state['outliers_handled'] = True
                    st.success("Outliers have been handled successfully.")
                    st.empty()
                    st.write(f"Number of outliers in {column_for_outlier} = 0")
                    # Visualize the data after handling based on the selected method
                    fig_after, ax_after = plt.subplots()
                    sns.boxplot(x=df[column_for_outlier], ax=ax_after)
                    st.subheader(f"After Handling Outliers ({outlier_method.capitalize()})")
                    st.pyplot(fig_after)
                    reset_all_flags()
        
    
        if 'outliers_handled' in st.session_state and st.session_state['outliers_handled']:
            st.header("Data after Handling Outliers")
            st.write(df)
            st.session_state['outliers_handled'] = False

        column_to_visualize = st.sidebar.selectbox("Select Column for Visualization", df.columns, key="visualize_col")
        if st.sidebar.button("Visualize Data", key='visualize_data_btn'):
            reset_all_flags()
            st.session_state['visualize_data_run'] = True

        if 'visualize_data_run' in st.session_state and st.session_state['visualize_data_run']:
            st.header("Data Visualization")
            fig1, fig2 = visualize_data(df, column_to_visualize)
            st.pyplot(fig1)
            st.pyplot(fig2)
            st.session_state['visualize_data_run'] = False

        if st.sidebar.button("Correlation Matrix", key='correlation_btn'):
            reset_all_flags()
            st.session_state['correlation_run'] = True

        if 'correlation_run' in st.session_state and st.session_state['correlation_run']:
            st.header("Correlation Matrix")
            fig = correlation_matrix(df)
            if fig is not None:
                st.pyplot(fig)
            st.session_state['correlation_run'] = False

        if st.sidebar.button("Download dataset", key='download_btn'):
            download_dataset(df)

# --- Chat Application Section ---
def chat_application():
    st.title("🧠 Chat with Llama3.2 Locally")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi there! How can I assist you today?"}
        ]

    if "history" not in st.session_state:
        st.session_state["history"] = [
            {"role": "system", "content": "You are a helpful assistant that answers users' questions."}
        ]

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # For storing user and assistant exchanges

    # Display messages in chat
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
    show_history = st.sidebar.button("📜 Show Chat History")
    if show_history:
        st.sidebar.write("### Chat History")
        for chat in reversed(st.session_state["chat_history"]):  # Show newest first
            st.sidebar.write(f"**🧑 User**: {chat['user']}")
            st.sidebar.write(f"**🧠 Assistant**: {chat['ollama']}")
            st.sidebar.write("---")
    # Handle user input
    if prompt := st.chat_input():
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.session_state["history"].append({"role": "human", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Initialize the ChatOllama model
        llm = ChatOllama(
            model="llama3.2",
            temperature=0,
        )

        # Add a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

        # Stream the assistant's response
        response_content = ""
        stream = llm.stream(st.session_state["history"])
        for chunk in stream:
            # Extract text content from AIMessageChunk
            response_content += chunk.content
            # Dynamically update the placeholder text
            response_placeholder.markdown(response_content)

        # Save and display the final response
        st.session_state["messages"].append({"role": "assistant", "content": response_content})
        st.session_state["history"].append({"role": "assistant", "content": response_content})

        # Append to chat history for the sidebar
        st.session_state["chat_history"].append(
            {"user": prompt, "ollama": response_content}
        )


def chat_with_rag(dff):
    st.subheader("Chatusing RAG")

    def ollama_generate(query: str, model: str = "llama3.2:latest") -> str:
        """Generate a response using Ollama."""
        try:
            result = ollama.chat(model=model, messages=[{"role": "user", "content": query}])
            return result.get("message", {}).get("content", "No response content.")
        except Exception as e:
            return f"Error: {e}"

    # Function to chat with CSV using Ollama
    def chat_with_csv_ollama(df, prompt, model="llama3.2:latest", max_rows=10):
        """Chat with a CSV using Ollama."""
        # Summarize dataset: Include column names, row count, and sample rows
        summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        column_info = "Columns:\n" + "\n".join([f"- {col} (type: {str(df[col].dtype)})" for col in df.columns])
        sample_data = f"Sample rows:\n{df.head(5).to_string(index=False)}"

        # Include data content (limit rows if necessary)
        data_content = f"The dataset:\n{df.head(max_rows).to_string(index=False)}"

        # Create the query
        query = f"""
        You are a data assistant. Here is the summary of the dataset:
        {summary}
        {column_info}
        {sample_data}

        {data_content}

        Based on this dataset, answer the following question:
        {prompt}
        """
        
        # Use the ollama_generate function to get the response
        return ollama_generate(query, model=model)

    # Initialize session state for query and response history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []  # Stores the history as a list of dictionaries with roles and messages

    # App title
    st.title("ChatCSV powered by Ollama")

    # Upload CSV section
        

    if dff is not None:
        # Read and display the CSV
        st.info("CSV Uploaded Successfully")
        
        st.dataframe(dff, use_container_width=True)

        # Chat interface
        st.info("Chat Below")
        user_input = st.chat_input("Ask a question:")

        if user_input:
            # Add user query to the conversation
            st.session_state.conversation.append({"role": "user", "content": user_input})

            # Generate response from Ollama
            with st.spinner("Generating response..."):
                assistant_response = chat_with_csv_ollama(dff, user_input)

            # Add assistant response to the conversation
            st.session_state.conversation.append({"role": "assistant", "content": assistant_response})

        # Display the conversation
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["content"])
            elif message["role"] == "assistant":
                # Check if the message contains code blocks
                if "" in message["content"]:
                    # Split by code blocks
                    code_blocks = message["content"].split("")
                    for i, block in enumerate(code_blocks):
                        if i % 2 == 1:  # Odd indices are code blocks
                            st.code(block.strip(), language="python")  # Render as code
                        else:
                            if block.strip():  # Avoid rendering empty text
                                st.chat_message("assistant").markdown(block.strip())
                else:
                    st.chat_message("assistant").markdown(message["content"])

# --- Main Application ---
def main():
    
    st.sidebar.title("Your Data Quality Analyzer & AI Chat Assistant")
    # option = st.sidebar.button("Chat Using RAG")
    
    # if option == "Data Quality Analysis":
        
    
    data_quality_analysis()

if __name__ == "__main__":
    main()
