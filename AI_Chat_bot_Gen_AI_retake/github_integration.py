import os
from typing import Dict, Optional
from github import Github
import streamlit as st
from config import GITHUB_TOKEN, GITHUB_REPO, COMPANY_INFO

class GitHubIntegration:
    def __init__(self):
        self.github_token = GITHUB_TOKEN
        self.repo_name = GITHUB_REPO
        self.github_client = None
        self.repository = None
        
        if self.github_token:
            try:
                self.github_client = Github(self.github_token)
                self.repository = self.github_client.get_repo(self.repo_name)
            except Exception as e:
                st.error(f"Error connecting to GitHub: {str(e)}")
    
    def is_configured(self) -> bool:
        """Check if GitHub integration is properly configured."""
        return (
            self.github_token is not None and 
            self.github_client is not None and 
            self.repository is not None
        )
    
    def create_support_ticket(self, ticket_data: Dict) -> Optional[str]:
        """Create a support ticket as a GitHub issue."""
        if not self.is_configured():
            return None
            
        try:
            # Validate required fields
            required_fields = ['title', 'description', 'user_name', 'user_email']
            for field in required_fields:
                if not ticket_data.get(field):
                    st.error(f"Missing required field: {field}")
                    return None
            
            # Create issue title
            issue_title = f"[Support] {ticket_data['title']}"
            
            # Create issue body
            issue_body = self._format_issue_body(ticket_data)
            
            # Create labels
            labels = ["support", "customer-issue"]
            
            # Create the GitHub issue
            issue = self.repository.create_issue(
                title=issue_title,
                body=issue_body,
                labels=labels
            )
            
            return issue.html_url
            
        except Exception as e:
            st.error(f"Error creating GitHub issue: {str(e)}")
            return None
    
    def _format_issue_body(self, ticket_data: Dict) -> str:
        """Format the ticket data into a GitHub issue body."""
        body = f"""## Customer Support Ticket

### Customer Information
- **Name:** {ticket_data['user_name']}
- **Email:** {ticket_data['user_email']}

### Issue Summary
**{ticket_data['title']}**

### Detailed Description
{ticket_data['description']}

### Additional Information
- **Company:** {COMPANY_INFO['name']}
- **Contact:** {COMPANY_INFO['phone']} | {COMPANY_INFO['email']}
- **Created via:** AI Chatbot Support System

---
*This ticket was automatically created by the customer support AI system.*
"""
        return body
    
    def get_recent_issues(self, limit: int = 5) -> list:
        """Get recent support issues from the repository."""
        if not self.is_configured():
            return []
            
        try:
            issues = []
            for issue in self.repository.get_issues(state='open', labels=['support']):
                if len(issues) >= limit:
                    break
                    
                issues.append({
                    'number': issue.number,
                    'title': issue.title,
                    'state': issue.state,
                    'created_at': issue.created_at,
                    'url': issue.html_url
                })
            
            return issues
            
        except Exception as e:
            st.error(f"Error fetching recent issues: {str(e)}")
            return []
    
    def test_connection(self) -> bool:
        """Test the GitHub connection and repository access."""
        if not self.is_configured():
            return False
            
        try:
            # Try to access repository information - check for common branch names
            branch_names = ["main", "master", "develop"]
            branch_found = False
            
            for branch_name in branch_names:
                try:
                    repo_info = self.repository.get_branch(branch_name)
                    branch_found = True
                    st.info(f"✅ Found branch: {branch_name}")
                    break
                except Exception as branch_error:
                    if "Branch not found" in str(branch_error):
                        continue
                    else:
                        st.error(f"Error checking branch {branch_name}: {str(branch_error)}")
                        return False
            
            if not branch_found:
                st.warning("⚠️ No common branches (main/master/develop) found")
                st.info("Repository exists but may have a different default branch")
                # Try to get repository info instead
                repo_info = self.repository.get_branch(self.repository.default_branch)
                st.info(f"✅ Using default branch: {self.repository.default_branch}")
                return True
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Branch not found" in error_msg:
                st.error("GitHub connection test failed: Repository exists but branch not found")
                st.info("Please check your GITHUB_REPO setting and ensure the repository has a main/master branch")
            else:
                st.error(f"GitHub connection test failed: {error_msg}")
            return False

