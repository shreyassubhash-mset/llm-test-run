import subprocess
import requests

def pushed_code():
    try:
        # Get the new code from the git repository
        new_code_bytes = subprocess.check_output(['git', 'show', 'HEAD...origin/main'])
        new_code = new_code_bytes.decode('utf-8')

        # Replace line breaks with '\n' 
        new_code_formatted = new_code.replace('\r\n', '\n').replace('\r', '\n')

        # Send the formatted code for review
        response = requests.post('http://localhost:8000/review', json={'code': new_code_formatted})
        if response.status_code == 200:
            review_result = response.json()
            print(review_result)
            return review_result
        else:
            print(f"Failed to get code review: {response.status_code}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Failed to get git commit details: {e}")
        return None

if __name__ == "__main__":
    pushed_code()