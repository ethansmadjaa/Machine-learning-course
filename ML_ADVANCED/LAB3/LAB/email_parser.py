import os

def parse_emails(directory):
    """Parse emails from a directory and categorize as spam or ham based on filename prefix
    
    Args:
        directory (str): Directory containing email files
        
    Returns:
        tuple: (ham_emails, spam_emails) lists of file paths
    """
    ham_emails = []
    spam_emails = []
    
    # Get all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Check if it's spam or ham based on filename prefix
        # In the ling-spam dataset, filenames starting with 'spmsg' are spam
        if filename.startswith('spmsg'):
            spam_emails.append(filepath)
        else:
            ham_emails.append(filepath)
    
    return ham_emails, spam_emails
