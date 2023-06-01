import requests


def send_data(message):
    # Send a POST request to the server
    response = requests.post(url, json=message)
    if response.status_code == 200:
        result = response.json()
        print(result['response'])
    else:
        print('Error:', response.status_code)


def client_program():
    message = ""  # take input
    while message.lower().strip() != 'bye':

        message = input(" -> ")
        data = {'message': message}
        send_data(data)


if __name__ == '__main__':
    url = 'http://localhost:5000/process'
    client_program()
