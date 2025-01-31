import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import openai

import JiraService, campaign_class
import sql_conn
from campaign_class import Campaign

st.set_page_config(layout="wide")

# Use CSS to remove top space
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("userdata.xlsx")  # Update with your file path
    return df

df = load_data()

# Left Sidebar: User Database
with st.sidebar:
    st.header("User Database Filters")
    # Columns to exclude from filters
    excluded_columns = ["USER_ID", "CLIENT_NAME", "AGE", "INTERNET_USAGE", "SMS_USAGE",
                        "CALLS_USAGE", "REVENUE", "PHONE"]

    filters = {}
    for col in df.columns:
        if col not in excluded_columns:  # Exclude specific columns
            unique_values = df[col].dropna().unique()  # Get unique non-null values
            if len(unique_values) > 1:  # Only add dropdown if multiple values exist
                filters[col] = st.multiselect(f"Select {col}", options=unique_values)

    # Apply filtering
    filtered_df = df.copy()
    for col, selected_values in filters.items():
        if selected_values:
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]


st.title("Marketing Campaign Management")
# Central Panel: Marketing Campaign Management
db_info_col, graph_col, ai_chat_col = st.columns([1, 1, 1])
# Display dataset information
with db_info_col:
# Target Segment Selection
    st.subheader("Dataset Information")

    # Calculate required statistics
    total_rows = len(filtered_df)
    avg_internet_usage = filtered_df["INTERNET_USAGE"].mean()
    avg_sms_usage = filtered_df["SMS_USAGE"].mean()
    avg_calls_usage = filtered_df["CALLS_USAGE"].mean()
    avg_revenue = filtered_df["REVENUE"].mean()

    # Display summary
    st.metric("Total Number of Users", total_rows)
    st.metric("Average Internet Usage", f"{avg_internet_usage:.2f} MB")
    st.metric("Average SMS Usage", f"{avg_sms_usage:.2f} messages")
    st.metric("Average Calls Usage", f"{avg_calls_usage:.2f} minutes")
    st.metric("Average Revenue", f"{avg_revenue:.2f} UZS")

    # product_name = st.text_input("Enter Product Name", key="product_name")
    # st.button("Create Campaign")
with graph_col:
# Target Segment Selection
    st.subheader("User Distribution")
    selected_param = st.selectbox("Select Parameter for Pie Chart", filters)

    # Filter data based on selected user database filters
    selected_param_data = filtered_df[selected_param]
    pie_chart = px.pie(selected_param_data, names=selected_param)
    st.plotly_chart(pie_chart)


# OpenAI API Key (Replace with your actual key)
OPENAI_API_KEY = ""

# Function to call ChatGPT and generate SMS message
def generate_sms(user_prompt, selected_filters):
    openai.api_key = OPENAI_API_KEY
    prompt = f"""Generate a short SMS message (max 160 characters) for a marketing campaign for the mobile operator company.
    This is the user prompt '{user_prompt}' that describes product to be offered and the target segment is based on these filters: {selected_filters}.
    Format the response as follows:
    Campaign name: [You give the name for the product] 
    Segment: {selected_filters}
    Message: [Your generated SMS message]
    
    "Campaign name:", "Segment:" and "Message:" should be in bold and each of it should start from new line"""
    print(f"This is the user prompt: {user_prompt}")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use a suitable GPT model
        messages=[{"role": "system", "content": "You are an expert SMS marketer."},
                  {"role": "user", "content": prompt}],
        max_tokens=200
    )

    return response.choices[0].message.content


# Function to call ChatGPT and generate SMS message
def create_campaign_function_execution():
    database_schema_string = """
                            Table: campaigns
                            Columns: campaign_id, campaign_type, campaign_date, campaign_name, segment, campaign_text, campaign_jiratask
                            """
    prompt = f"""Use this function to add a new campaign to the database. Output should be a fully formed SQL query.
                This is database schema: {database_schema_string}
                This is the chat history: {st.session_state.chat_history}
                Take last response from AI as campaign parameters, use DATE('now') for today's date, do not put any quotation marks in the query
                """

    print(f"This is function prompt for sql write: {prompt}")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use a suitable GPT model
        messages=[{"role": "system", "content": "You are SQL script writer."},
                  {"role": "user", "content": prompt}],
        max_tokens=200
    )
    print(f"This is the script: {response.choices[0].message.content}")
    return sql_conn.add_campaign(response.choices[0].message.content)


# Right Sidebar: AI Chatbot
with ai_chat_col:
    st.subheader("SMS Text Generator AI")
    # Placeholder for dynamically updating chat history
    chat_placeholder = st.empty()
    # Conversation history storage
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User Input for product
    user_prompt = st.text_input("You can ask to create a product/message for selected segment", key="chat_product_name")

    # Extract selected filter values
    selected_filters = {col: val for col, val in filters.items() if val}

    send_button, create_button = st.columns([1, 1])
    with send_button:
        if st.button("Send"):
            if user_prompt:
                # Extract product name and filters from user input
                selected_filters_str = ", ".join([f"{k}: {v}" for k, v in selected_filters.items()])

                # Append user message to chat history
                st.session_state.chat_history.append({"role": "User", "message": user_prompt})

                # Generate SMS message.
                sms_response = generate_sms(st.session_state.chat_history, selected_filters_str)
                print(f"This is ChatGPT response: {sms_response}")
                # Clear the input field by setting it to an empty string
                st.session_state.user_input = ""
                # Append AI response to chat history
                st.session_state.chat_history.append({"role": "AI", "message": sms_response})

            # Dynamically update chat history in the container
    with chat_placeholder.container():
        for chat in st.session_state.chat_history:
            with st.chat_message("user" if chat["role"] == "User" else "assistant"):
                st.write(chat["message"])
    with create_button:
        if st.button("Create campaign"):
            create_campaign_function_execution()
            st.write(JiraService.create_jira_task(Campaign.get_latest_campaign()))

# Campaign Calendar
st.subheader("Campaign Calendar")
# Display ongoing or accomplished campaigns for selected date
campaigns_data = sql_conn.get_campaigns()
# Allow user to select multiple dates
# Convert CAMPAIGN_DATE to datetime format
campaigns_data["CAMPAIGN_DATE"] = pd.to_datetime(campaigns_data["CAMPAIGN_DATE"]).dt.date

# Extract available campaign dates
available_dates = sorted(campaigns_data["CAMPAIGN_DATE"].unique())


# Allow user to select date range
selected_dates_range = st.date_input(
    "Select Date Range",
    value=[available_dates[0], available_dates[-1]],  # Default range to include all available dates
    min_value=min(available_dates),
    max_value=max(available_dates),
)

# Ensure selected_dates_range is always a list (even if only one date is selected)
if isinstance(selected_dates_range, datetime.date):  # If only one date is selected
    selected_dates_range = [selected_dates_range]

if len(selected_dates_range)<2:
    # Create a date range that includes all days between the selected start and end dates
    start_date = selected_dates_range[0]
    end_date = selected_dates_range[0]
else:
    start_date = selected_dates_range[0]
    end_date = selected_dates_range[1]

date_range = pd.date_range(start=start_date, end=end_date).date

# Filter campaigns based on the selected date range
filtered_campaigns = campaigns_data[campaigns_data["CAMPAIGN_DATE"].isin(date_range)]

# Display campaigns in a table
if not filtered_campaigns.empty:
    st.dataframe(filtered_campaigns, height=300)  # Displays as a scrollable table
else:
    st.write("No campaigns found for selected dates.")