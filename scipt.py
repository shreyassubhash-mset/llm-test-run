import subprocess
import requests

def pushed_code():
    try:
        new_code = subprocess.check_output(['git','show', 'HEAD...origin/main']).decode('utf-8')
        
        # Replace line breaks with '\n' 
        new_code_formatted = new_code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Send the formatted code for review
        response = requests.post('http://localhost:8000/review', json={'code': new_code_formatted})
        if response.status_code == 200:
            review_result = response.json().get('review', [])
            concatenated_data = ''
           # Iterate through each string in the array and concatenate it to the result string
            for string in review_result:
                concatenated_data += string

            print(string)
            return string
        else:
            print(f"Failed to get code review: {response.status_code}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Failed to get git commit details: {e}")
        return None

if __name__ == "__main__":
    pushed_code()