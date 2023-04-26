from validate_email import validate_email
is_valid = validate_email(email_address='sharathhebbar@gmail.com',   
    smtp_timeout=10, dns_timeout=10)
print(is_valid)