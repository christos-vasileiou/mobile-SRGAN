import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_email(to='chrivasileiou2@gmail.com', subject='', html_content=''):
    message = Mail(
        from_email='chrivasileiou2@gmail.com',
        to_emails=to,
        subject=subject,
        html_content=html_content)
    try:
        sg = SendGridAPIClient(os.environ.get('EMAILKEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)
