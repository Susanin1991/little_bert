import requests

# Set the server URL
server_url = 'http://169.255.254.2:5000/process'

def send_data(data):
    # Send a POST request to the server
    response = requests.post(server_url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(result['message'])
    else:
        print('Error:', response.status_code)

# Example usage
data = {'name': 'John', 'age': 30}
send_data(data)