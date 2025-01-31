GENERAL
1. The solution is for creating marketing campaigns like SMS for telecom users. Real life case.
2. There are two datasources used - campaigns.db, which stores all marketing campaigns and userdata.xlsx for user database
3. UI and Campaigns_table png images show the screenshots of implemented solution
4. Installation and running requires - GPT API key and JIRA credentials (in case of testing, I can provide one)
5. Requirements for libraries are listed out in requirements.txt

UI AND LOGIC 

There is a left sidebar where filters for user database are placed. 
The user database can be filtered with the chosen parameters. 
In the middle section, information about the dataset along with pie chart for the user database after applying filters can be seen.
On the right panel, there is a chat where it can be asked to create a text from ChatGPT.
The chat with GPT is in conversational style and it can be asked from GPT to make corrections for the text.
Moreover, there is a "Create Campaign" button, which serves to ask from GPT to generate a sql insert function for the developed marketing campaign along with all parameters.
This insert query goes to the database and is commited. The campaign is saved.
After that, Jira task (JiraService) is created via API call for the technical team who execute the campaign.
At the end of the page, there is a table that shows all marketing campaigns with the ability to select range of dates.
Logs such as user_prompts, responses from GPT, etc. are printed on the console.

SECURITY COUNTERMEASURES

This application in real life should work with user database and user sensitive information. 

In order to prevent data leakage, several security measures can be applied:
1. Removing or partially hiding the user names, phone numbers, addresses.
2. Not allow AI to access columns of sensitive data in the tables
3. Not allow AI update, insert queries (even though I have used this on my solution, haha).
4. Not allow AI to get all database without any filters
5. Check each query generated (if there will be any) for specific query keywords such as "drop", "insert", "update", etc.